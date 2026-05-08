from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
APP = ROOT / "app.py"
URL = "http://127.0.0.1:8501"


def wait_for_server(timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    return False


def main() -> int:
    if not PYTHON.exists():
        print(f"Could not find virtual environment Python: {PYTHON}")
        return 1
    if not APP.exists():
        print(f"Could not find Streamlit app: {APP}")
        return 1

    command = [
        str(PYTHON),
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
    ]

    log_path = ROOT / "streamlit_server.log"
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    print(f"Started Streamlit server process: {process.pid}")
    print(f"Log file: {log_path}")
    print("Waiting for server health check...")

    if wait_for_server():
        print(f"Dashboard is ready: {URL}")
        subprocess.Popen(["cmd", "/c", "start", "", URL])
        print("Keep this window open. Press Ctrl+C here only when you want to stop the dashboard.")
        try:
            return process.wait()
        except KeyboardInterrupt:
            print("\nStopping dashboard...")
            process.terminate()
            return 0

    print("Dashboard did not become ready. Recent log output:")
    log.flush()
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-80:]))
    process.terminate()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
