"""Menghitung embedding lokal dengan seluruh koneksi jaringan diblokir."""
from __future__ import annotations

import os
import runpy
import sys


def install_offline_guard():
    for name, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "ANONYMIZED_TELEMETRY": "false", "OTEL_SDK_DISABLED": "true",
        "OTEL_TRACES_EXPORTER": "none", "OTEL_METRICS_EXPORTER": "none",
        "OTEL_LOGS_EXPORTER": "none",
    }.items():
        os.environ[name] = value

    def deny_network(event, arguments):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo",
                     "socket.sendto", "http.client.connect", "urllib.Request"}:
            raise PermissionError("Jaringan diblokir untuk proses embedding offline.")

    sys.addaudithook(deny_network)


def main():
    install_offline_guard()
    runpy.run_module("scripts.cache_evaluation_embeddings", run_name="__main__")


if __name__ == "__main__":
    main()
