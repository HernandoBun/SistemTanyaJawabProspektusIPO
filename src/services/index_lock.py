"""Coordinate index writers across application sessions and CLI processes."""

from contextlib import contextmanager

from filelock import FileLock, Timeout

import config


class IndexWriteBusyError(RuntimeError):
    """The shared indexes already have an active writer."""


@contextmanager
def index_write_lock():
    lock = FileLock(config.CACHE_REGISTRY_PATH.with_suffix(".lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise IndexWriteBusyError(
            "Sistem sedang memproses perubahan indeks. Coba lagi setelah proses tersebut selesai."
        ) from exc
    try:
        yield
    finally:
        lock.release()
