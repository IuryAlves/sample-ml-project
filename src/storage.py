import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import joblib
import pandas as pd
from google.api_core.exceptions import BadRequest, NotFound
from google.cloud.storage import Client
from google.resumable_media.common import InvalidResponse

logger = logging.getLogger(__name__)


class Storage:
    def __init__(self, *paths, path: Any = None, **options) -> "Self":
        self.path = list(paths) + [path]
        self.backend = _backend(path=self.path, **options)

    def copy(self, object: Any, path: Any = None) -> None:
        return self.backend.copy(object=object, path=self._locate(path))

    def describe(
        self,
        path: Any = None,
        absolute: bool = False,
    ) -> Iterator[Tuple[str, Any]]:
        return map(
            lambda pair: ((pair[0] if absolute else self._clean(pair[0])), pair[1]),
            self.backend.describe(path=self._locate(path)),
        )

    def exists(self, path: Any = None) -> bool:
        return self.backend.exists(path=self._locate(path))

    def file(self, path: Any = None) -> bool:
        return self.backend.file(path=self._locate(path))

    def list(self, path: Any = None, absolute: bool = False) -> Iterator[str]:
        return map(lambda pair: pair[0], self.describe(path=path, absolute=absolute))

    def locate(self, *path, absolute: bool = False) -> str:
        path = self.backend.locate(*_flatten([self.path, path]))
        return path if absolute else self._clean(path)

    def make(self, path: Any = None) -> None:
        return self.backend.make(path=self._locate(path))

    def read(self, path: Any = None, **options) -> Any:
        return self.backend.read(path=self._locate(path), **options)

    def write(self, object: Any, path: Any = None, **options) -> None:
        self.backend.write(object=object, path=self._locate(path), **options)

    def _clean(self, path: str) -> str:
        name = uuid.uuid4().hex
        path_ = self.backend.locate(*_flatten([self.path, name]))
        return path.replace(path_.replace(name, ""), "")

    def _locate(self, path: Any = None) -> str:
        return self.backend.locate(*_flatten([self.path, path]))


class _Base:
    def __init__(self, **options) -> "Self":
        self.options = options

    def copy(self, object: Any, path: str) -> None:
        _copy(Storage(object, **self.options).backend, object, self, path)


class _Cloud(_Base):
    PATTERN = "^(gs://|/gcs/)"
    PREFIX = "gs://"
    SEPARATOR = "/"

    @staticmethod
    def locate(*path) -> Optional[str]:
        path = _Cloud.SEPARATOR.join(path)
        return (
            re.sub(_Cloud.PATTERN, _Cloud.PREFIX, path)
            if re.match(_Cloud.PATTERN, path)
            else None
        )

    @staticmethod
    def split(path: str) -> Tuple[str, str, str]:
        path = re.sub(_Cloud.PATTERN, "", path).split(_Cloud.SEPARATOR)
        return path[0], _Cloud.SEPARATOR.join(path[1:]), path[-1]

    def __init__(self, *arguments, **options) -> "Self":
        super().__init__(*arguments, **options)
        self.buckets = {}
        self.clients = {}

    def bucket(self, name: str) -> "Bucket":
        if name not in self.buckets:
            self.buckets[name] = self.client().bucket(name)
        return self.buckets[name]

    def client(self, name: str = "default") -> Client:
        if name not in self.clients:
            self.clients[name] = Client(**self.options)
        return self.clients[name]

    def describe(self, path: str) -> Iterator[Tuple[str, Any]]:
        bucket, blob, _ = _Cloud.split(path)
        return (
            (self.SEPARATOR.join([f"{self.PREFIX}{bucket}", blob.name]), blob)
            for blob in self.client().list_blobs(bucket, prefix=blob)
        )

    def exists(self, path: str) -> bool:
        return next(self.describe(path)) is not None

    def file(self, path: str) -> bool:
        bucket, blob, _ = _Cloud.split(path)
        return self.bucket(bucket).blob(blob).exists()

    def make(self, path: str) -> None:
        pass

    def read(self, path: str, **options) -> Any:
        bucket, blob, name = _Cloud.split(path)
        blob = self.bucket(bucket).blob(blob)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, name)
            blob.download_to_filename(path)
            return _Local().read(path, **options)

    def write(self, object: Any, path: str, **options) -> None:
        _, _, name = _Cloud.split(path)
        with tempfile.TemporaryDirectory() as root:
            path_ = os.path.join(root, name)
            _Local().write(object, path_, **options)
            self.copy(object=path_, path=path)


class _Local(_Base):
    @staticmethod
    def locate(*path) -> Optional[str]:
        return os.path.join(*path) if path else "."

    def describe(self, path: str) -> Iterator[Tuple[str, Any]]:
        if not self.exists(path):
            return []
        if self.file(path):
            return [(path, None)]
        return (
            (os.path.join(path, root, name), None)
            for root, _, names in os.walk(path)
            for name in names
        )

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(path)

    @staticmethod
    def file(path: str) -> bool:
        return os.path.isfile(path)

    @staticmethod
    def make(path: str) -> None:
        if path:
            os.makedirs(path, exist_ok=True)

    def read(self, path: str, format: Optional[str] = None, **options) -> Any:
        format = self._format(path, format)
        if format == "csv":
            return pd.read_csv(path, **options)
        if format == "joblib":
            return joblib.load(path, **options)
        if format == "json":
            with open(path, encoding="utf-8") as file:
                return json.load(file, **options)
        if format == "text":
            with open(path, encoding="utf-8") as file:
                return file.read()
        assert False

    def write(
        self,
        object: Any,
        path: str,
        format: Optional[str] = None,
        **options,
    ) -> None:
        format = self._format(path, format)
        self.make(os.path.dirname(path))
        if format == "csv":
            object.to_csv(path, **options)
            return
        if format == "joblib":
            joblib.dump(object, path, **options)
            return
        if format == "json":
            with open(path, "w", encoding="utf-8") as file:
                json.dump(object, file, **options)
            return
        if format == "text":
            with open(path, "w", encoding="utf-8") as file:
                file.write(object)
            return
        assert False

    def _format(self, path: str, format: Optional[str] = None) -> str:
        mapping = dict(txt="text")
        format = format if format else os.path.splitext(path)[1].replace(".", "")
        format = format if format else "joblib"
        return mapping.get(format, format)


def copy(source: Any, destination: Any) -> None:
    Storage(destination).copy(source)


def download(
    paths: List[str],
    root: str,
    names: Optional[List[str]] = None,
    delete: bool = True,
    parallelize: bool = True,
    skip_failed: bool = False,
    **options,
) -> Sequence:
    """Download files from the cloud to the local machine.

    Arguments:
        paths: A list of files in the cloud.

        root: A directory on the local machine.

        names: A list of locations relative to `root` where the corresponding
        files should be saved. If not specified, the filenames will be used,
        losing the directory structure in the cloud.

        delete: A flag indicating if extraneous files should be deleted.

        parallelize: A flag indicating if parallelization should be used.

        skip_failed: A flag if the failed downloads should be skipped.

    All other arguments are forwarded to `parallelize` and `_download`.

    Returns:
        A list of absolute locations on the local machine.
    """
    if names is None:
        names = [path.split(_Cloud.SEPARATOR)[-1] for path in paths]
    if delete:
        names_ = [
            path.replace(os.path.join(root, ""), "")
            for path, _ in _Local().describe(root)
        ]
        names_ = set(names_).difference(set(names))
        if names_:
            logger.warning("Removing %d extraneous files...", len(names_))
        for name in names_:
            os.remove(os.path.join(root, name))
    pairs = list(zip(paths, names))
    return _download(pairs, root=root, skip_failed=skip_failed, **options)


def read(path: Any, **options) -> Any:
    return Storage(path).read(**options)


def synchronize(source: str, destination: str, delete: bool = True) -> None:
    source = re.sub(_Cloud.PATTERN, _Cloud.PREFIX, source)
    destination = re.sub(_Cloud.PATTERN, _Cloud.PREFIX, destination)
    Storage(destination).make()
    if delete:
        command = ["gsutil", "-m", "rsync", "-d", "-r", source, destination]
    else:
        command = ["gsutil", "-m", "rsync", "-r", source, destination]
    try:
        subprocess.check_output(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exception:
        raise SystemError(
            exception.output.decode(sys.getfilesystemencoding())
        ) from exception


def write(object: Any, path: Any, **options) -> None:
    Storage(path).write(object, **options)


def _backend(path: Any, **options) -> _Base:
    path = _flatten(path)
    if path:
        # Select a backend dynamically based on the given path.
        klasses = [_Cloud, _Local]
        klass = next(klass for klass in klasses if klass.locate(*path))
    elif options:
        # Default to the cloud backend if there are options given.
        klass = _Cloud
    else:
        # Default to the local backend otherwise.
        klass = _Local
    return klass(**options)


def _copy(
    source_backend: _Base,
    source_path: str,
    destination_backend: _Base,
    destination_path: str,
) -> None:
    globals()[
        "_copy"
        + source_backend.__class__.__name__.lower()
        + destination_backend.__class__.__name__.lower()
    ](
        source_backend,
        source_path,
        destination_backend,
        destination_path,
    )


def _copy_local_cloud(
    # pylint: disable=unused-argument
    source_backend: _Local,
    source_path: str,
    destination_backend: _Cloud,
    destination_path: str,
) -> None:
    bucket, blob, _ = _Cloud.split(destination_path)
    bucket = destination_backend.bucket(bucket)
    blob = bucket.blob(blob)
    blob.upload_from_filename(source_path)


def _copy_cloud_cloud(
    source_backend: _Cloud,
    source_path: str,
    destination_backend: _Cloud,
    destination_path: str,
) -> None:
    bucket, blob, _ = _Cloud.split(destination_path)
    bucket = destination_backend.bucket(bucket)
    blob = bucket.blob(blob)
    bucket_, blob_, _ = _Cloud.split(source_path)
    bucket_ = source_backend.bucket(bucket_)
    blob_ = bucket_.blob(blob_)
    bucket_.copy_blob(blob_, bucket, blob.name)


def _copy_cloud_local(
    source_backend: _Cloud,
    source_path: str,
    destination_backend: _Local,
    destination_path: str,
) -> None:
    bucket, blob, _ = _Cloud.split(source_path)
    bucket = source_backend.bucket(bucket)
    blob = bucket.blob(blob)
    destination_backend.make(os.path.dirname(destination_path))
    blob.download_to_filename(destination_path)


def _copy_local_local(
    # pylint: disable=unused-argument
    source_backend: _Local,
    source_path: str,
    destination_backend: _Local,
    destination_path: str,
) -> None:
    destination_backend.make(os.path.dirname(destination_path))
    shutil.copyfile(source_path, destination_path)


# pylint: disable=too-many-locals
def _download(
    pairs: List[Tuple[str, str]],
    root: str,
    project: Optional[str] = None,
    skip_failed: bool = False,
    attempts: int = 3,
    timeout: int = 1,
) -> List[str]:
    """Download files from the cloud to the local machine.

    Arguments:
        pairs: A list of pairs where the first is a file in the cloud, and the
        second is a location on the local machine relative to `root`.

        root: A directory on the local machine.

        project: A project on Google Cloud Platform.

        attempts: The number of download attempts to make per file.

        timeout: The timeout between attempts.

    Returns:
        A list of absolute locations of successful downloads on the local machine.
    """
    cloud = _Cloud(project=project)
    local = _Local()
    downloaded_paths = []
    for path, name in pairs:
        path_ = local.locate(root, name)
        if local.exists(path_):
            continue
        for i in range(attempts):
            try:
                _copy_cloud_local(cloud, path, local, path_)
                downloaded_paths.append(path_)
                break
            except (BadRequest, InvalidResponse, NotFound) as exception:
                logger.warning("Download unsuccessful for %s: %s", path, exception)
                if skip_failed:
                    logger.warning("Skipping missing file %s", path)
                    continue
                if i + 1 == attempts:
                    raise
                time.sleep(timeout)
    return downloaded_paths


def _flatten(path: Any) -> List[str]:
    if path is None:
        return []
    if isinstance(path, str):
        return [path]
    if isinstance(path, (list, tuple)):
        return [path__ for path_ in path for path__ in _flatten(path_)]
    assert False