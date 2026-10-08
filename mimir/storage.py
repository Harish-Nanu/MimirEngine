"""MinIO / S3-compatible object storage for raw documents."""

from __future__ import annotations

from collections.abc import Iterator
from typing import BinaryIO

from .config import Settings
from .errors import PermanentError, TransientError


class MinioStore:
    def __init__(self, settings: Settings):
        from minio import Minio

        self.client = Minio(
            settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_secure,
        )

    def ensure_bucket(self, bucket: str) -> None:
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def put(self, bucket: str, key: str, stream: BinaryIO, content_type: str | None) -> None:
        self.client.put_object(
            bucket,
            key,
            stream,
            length=-1,
            part_size=10 * 1024 * 1024,
            content_type=content_type or "application/octet-stream",
        )

    def get(self, bucket: str, key: str) -> bytes:
        from minio.error import S3Error

        try:
            resp = self.client.get_object(bucket, key)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchBucket"):
                raise PermanentError(f"object s3://{bucket}/{key} not found") from exc
            raise TransientError(f"S3 error reading s3://{bucket}/{key}: {exc.code}") from exc
        except OSError as exc:  # connection resets, timeouts (urllib3 errors subclass OSError)
            raise TransientError(f"network error reading s3://{bucket}/{key}: {exc}") from exc

    def list_keys(self, bucket: str, prefix: str = "") -> Iterator[str]:
        for obj in self.client.list_objects(bucket, prefix=prefix, recursive=True):
            if not obj.is_dir:
                yield obj.object_name
