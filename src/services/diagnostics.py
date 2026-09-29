"""Actionable errors for the offline CLI and online interface."""
import socket
from urllib.parse import urlsplit

import config


def check_embedding_connection() -> None:
    """Check configuration and DNS before expensive document processing."""
    if config.EMBEDDING_PROVIDER != "openrouter":
        return
    if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY.startswith("your_"):
        raise ValueError("Isi OPENROUTER_API_KEY pada file .env.")
    url = urlsplit(config.OPENROUTER_BASE_URL)
    if url.scheme != "https" or not url.hostname:
        raise ValueError("OPENROUTER_BASE_URL harus berupa alamat HTTPS yang valid.")
    try:
        socket.getaddrinfo(url.hostname, url.port or 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ConnectionError(
            f"Alamat server {url.hostname} tidak dapat ditemukan. "
            "Periksa koneksi internet, DNS, VPN/proxy, dan OPENROUTER_BASE_URL."
        ) from exc


def describe_error(error: Exception) -> str:
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & {"APIConnectionError", "ConnectError", "APITimeoutError", "TimeoutException"}:
        return ("Tidak dapat terhubung ke layanan API. Periksa internet, DNS, VPN/proxy, "
                "dan alamat layanan pada .env, lalu jalankan kembali. "
                "Hasil parsing yang sudah tersimpan dapat digunakan kembali.")
    if names & {"AuthenticationError", "PermissionDeniedError"}:
        return "Layanan menolak akses. Periksa API key dan izin layanan pada .env."
    if names & {"RateLimitError"}:
        return "Batas penggunaan layanan tercapai. Periksa kuota dan coba lagi setelah jeda."
    return str(error)
