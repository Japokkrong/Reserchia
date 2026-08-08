"""Persisting chat sessions: Postgres for threads, S3-compatible for files.

Everything here is **opt-in on `DATABASE_URL`**. Unset -- which is how the app
runs on the host, and how the CLI always runs -- and none of it activates: no
auth, no data layer, in-memory conversations exactly as before. That keeps the
Docker stack from becoming a requirement for running the project at all.

Two things worth knowing before changing any of it:

**Thread history requires authentication.** Chainlit's `get_current_user`
returns `None` when no auth callback is registered, and threads are owned by a
user identifier -- so with no auth there is nothing to own them and the sidebar
stays empty. `header_auth_callback` satisfies that transparently, with no login
page, which is only acceptable because compose publishes the port on 127.0.0.1.

**RustFS starts with no buckets.** The first element upload would 404 against a
missing bucket, so it is created at startup rather than assumed.
"""

from __future__ import annotations

import os

#: Files Chainlit uploads and generates -- diagram PNGs, citation passages.
#: Not the papers, and not the library.
DEFAULT_BUCKET = "reserchia"

#: A single local user. Compose binds the app to localhost, so this identifies
#: rather than authenticates: it exists to give threads an owner.
LOCAL_USER = "local"


def _public_storage_class():
    """Storage client that signs read URLs for a host the browser can reach.

    The problem it solves is specific to running the store in Docker. The app
    uploads over the compose network, so its endpoint is `http://rustfs:9000`.
    But `get_read_url` returns a **presigned URL that the browser fetches
    directly**, and `rustfs` is not a name any browser can resolve -- the page
    fills with `ERR_NAME_NOT_RESOLVED` while uploads look perfectly healthy.

    So reads are signed by a second client aimed at the published address.
    Signing against the right host, rather than rewriting the host afterwards,
    is what keeps this correct: today RustFS gets SigV2 URLs, where the host is
    not part of the signed string and a rewrite would happen to work, but SigV4
    signs the host and a rewrite would silently start failing.
    """
    import boto3
    from chainlit.data.storage_clients.s3 import S3StorageClient

    class _PublicUrl(S3StorageClient):
        def __init__(self, bucket: str, public_endpoint: str, credentials: dict, **kwargs):
            super().__init__(bucket=bucket, **kwargs)
            self._public = boto3.client(
                "s3", endpoint_url=public_endpoint, **credentials
            )

        def sync_get_read_url(self, object_key: str) -> str:
            from chainlit.data.storage_clients.base import storage_expiry_time

            try:
                return self._public.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": object_key},
                    ExpiresIn=storage_expiry_time,
                )
            except Exception:  # noqa: BLE001 - a missing image is not fatal
                return object_key

    return _PublicUrl


class _LazyPublicClient:
    """Defers the boto3/chainlit import until a client is actually built."""

    def __call__(self, **kwargs):
        return _public_storage_class()(**kwargs)


_PublicUrlStorageClient = _LazyPublicClient()


def database_url() -> str:
    """The Postgres DSN, or "" when persistence is off."""
    return os.getenv("DATABASE_URL", "").strip()


def enabled() -> bool:
    return bool(database_url())


def _storage_client():
    """S3 client pointed at RustFS, or None if it is not configured.

    Chainlit's `S3StorageClient` forwards `**kwargs` straight to
    `boto3.client("s3", ...)`, so `endpoint_url` is all it takes to aim a client
    written for AWS at RustFS or MinIO.
    """
    endpoint = os.getenv("RUSTFS_ENDPOINT", "").strip()
    if not endpoint:
        return None

    bucket = os.getenv("RUSTFS_BUCKET", DEFAULT_BUCKET).strip() or DEFAULT_BUCKET
    credentials = {
        "aws_access_key_id": os.getenv("RUSTFS_ACCESS_KEY", ""),
        "aws_secret_access_key": os.getenv("RUSTFS_SECRET_KEY", ""),
        "region_name": os.getenv("RUSTFS_REGION", "us-east-1"),
    }
    client = _PublicUrlStorageClient(
        bucket=bucket,
        public_endpoint=os.getenv("RUSTFS_PUBLIC_ENDPOINT", "").strip() or endpoint,
        credentials=credentials,
        endpoint_url=endpoint,
        **credentials,
    )
    _ensure_bucket(client, bucket)
    return client


def _ensure_bucket(client, bucket: str) -> None:
    """Create the bucket and allow the browser to read from it.

    Both halves are needed on a fresh volume, and both fail in ways that look
    like something else:

    - **No bucket**: the first upload 404s.
    - **No CORS rule**: uploads succeed and the page stays blank, because the
      browser fetches presigned URLs from `localhost:19000` while the app is
      served from `localhost:18000` -- a cross-origin request that RustFS
      rejects by default with no `Access-Control-Allow-Origin`.

    Doing it here rather than in an init container keeps a fresh volume
    self-healing instead of leaving a manual step in the README.
    """
    try:
        boto = client.client
        existing = {b["Name"] for b in boto.list_buckets().get("Buckets", [])}
        if bucket not in existing:
            boto.create_bucket(Bucket=bucket)

        boto.put_bucket_cors(
            Bucket=bucket,
            CORSConfiguration={
                "CORSRules": [
                    {
                        "AllowedHeaders": ["*"],
                        "AllowedMethods": ["GET", "HEAD"],
                        # The store is bound to localhost, so the origin that
                        # can reach it is already restricted by the network.
                        "AllowedOrigins": ["*"],
                        "ExposeHeaders": ["ETag"],
                        "MaxAgeSeconds": 3000,
                    }
                ]
            },
        )
    except Exception as exc:  # noqa: BLE001 - storage is not worth a hard failure
        # An unreachable store costs images in reopened threads, not answers.
        print(f"reserchia: could not prepare bucket {bucket!r}: {exc}")


def _columns(tables: dict) -> dict[str, set[str]]:
    """Actual column names per table, read with asyncpg.

    Runs in its own thread with its own event loop: `data_layer()` may be called
    from inside a running loop, where `asyncio.run` would raise, and asyncpg is
    already a dependency -- adding a synchronous driver just for a startup check
    would not be worth it.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    import asyncpg

    async def read() -> dict[str, set[str]]:
        # asyncpg speaks the plain DSN, not SQLAlchemy's dialect prefix.
        conn = await asyncpg.connect(
            database_url().replace("postgresql+asyncpg://", "postgresql://")
        )
        try:
            rows = await conn.fetch(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_name = ANY($1::text[])",
                list(tables),
            )
        finally:
            await conn.close()
        found: dict[str, set[str]] = {}
        for row in rows:
            found.setdefault(row["table_name"], set()).add(row["column_name"])
        return found

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(read())).result(timeout=15)


def check_schema() -> list[str]:
    """Report columns Chainlit will write that the database does not have.

    Worth doing at startup because the failure mode is silent. The data layer
    builds each INSERT from whatever keys the step or element carries, so a
    missing column aborts the write -- and it is caught and logged as a warning,
    never surfaced. The app looks healthy, answers normally, and simply stores
    no history. That is exactly how `autoCollapse` went unnoticed.

    Returns a list of "table.column" strings; empty means the schema is current.
    """
    if not enabled():
        return []

    from chainlit.element import ElementDict
    from chainlit.step import StepDict

    expected = {
        # `feedback` lives in its own table; everything else in these dicts is
        # written verbatim as a column.
        "steps": set(StepDict.__annotations__) - {"feedback"},
        "elements": set(ElementDict.__annotations__),
    }

    try:
        have = _columns(expected)
    except Exception as exc:  # noqa: BLE001 - a check must not block startup
        print(f"reserchia: could not verify the chat schema: {exc}")
        return []

    missing = [
        f"{table}.{column}"
        for table, columns in expected.items()
        for column in sorted(columns - have.get(table, set()))
    ]

    if missing:
        print(
            "reserchia: chat history will NOT be saved -- the database is "
            f"missing {', '.join(missing)}.\n"
            "  Apply: docker compose exec -T postgres psql -U reserchia "
            "-d reserchia < docker/migrate.sql"
        )
    return missing


def data_layer():
    """The Chainlit data layer, or None when persistence is off."""
    if not enabled():
        return None

    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    check_schema()
    return SQLAlchemyDataLayer(
        conninfo=database_url(),
        storage_provider=_storage_client(),
    )
