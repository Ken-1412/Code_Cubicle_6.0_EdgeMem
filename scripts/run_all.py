"""Launch a fresh two-device EdgeMem demo and its local SQLite cloud."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_ROOT = ROOT / "data" / "run"
SEED_FILE = ROOT / "data" / "seed_logs.jsonl"
DEVICE_URLS = {
    "device_a": "http://127.0.0.1:8001",
    "device_b": "http://127.0.0.1:8002",
}
CLOUD_URL = "http://127.0.0.1:9000"
SERVICE_PORTS = {"device_a": 8001, "device_b": 8002, "cloud": 9000}
BIND_HOST = os.environ.get("EDGEMEM_BIND_HOST", "127.0.0.1")


def create_device_app_from_environment():
    """Uvicorn factory used by each isolated device subprocess."""
    from edgemem.api import create_app

    try:
        db_path = os.environ["EDGEMEM_RUN_DB"]
        device_id = os.environ["EDGEMEM_RUN_DEVICE_ID"]
    except KeyError as error:
        raise RuntimeError(f"missing runner environment setting: {error.args[0]}") from error
    return create_app(
        db_path=db_path,
        device_id=device_id,
        cloud_url=os.environ.get("EDGEMEM_CLOUD_URL", CLOUD_URL),
    )


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.25)
        return client.connect_ex(("127.0.0.1", port)) == 0


def _preflight_ports() -> None:
    busy = [f"{name} ({port})" for name, port in SERVICE_PORTS.items() if _port_is_open(port)]
    if busy:
        raise RuntimeError("required demo ports are already in use: " + ", ".join(busy))


def _http_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | list[Any]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + path,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {request.full_url} returned {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"request failed for {request.full_url}: {error}") from error


def _wait_ready(name: str, process: subprocess.Popen, url: str, timeout_s: float = 45.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"{name} exited during startup with code {exit_code}")
        try:
            _http_json(url, "/health", timeout=1.0)
            print(f"ready: {name} {url}", flush=True)
            return
        except (RuntimeError, TimeoutError, OSError) as error:
            last_error = error
            time.sleep(0.25)
    raise TimeoutError(f"{name} did not become ready at {url}: {last_error}")


def _ingest_devices() -> None:
    """Use Module 4's CLI to seed two fresh local device stores."""
    for device_id in DEVICE_URLS:
        database = RUN_ROOT / device_id
        environment = os.environ.copy()
        environment["EDGEMEM_CLOUD_URL"] = CLOUD_URL
        command = [
            sys.executable,
            "-m",
            "edgemem.ingest",
            "--db",
            str(database),
            "--file",
            str(SEED_FILE),
            "--device-id",
            device_id,
            "--reset",
        ]
        print(f"seeding {device_id} with Module 4 ingestion...", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _queue_seed_snapshots() -> int:
    """Queue full canonical seed snapshots so first sync bootstraps cloud state."""
    from edgemem.changelog import ChangeLog
    from edgemem.store import LocalStore

    queued = 0
    for device_id in DEVICE_URLS:
        database = RUN_ROOT / device_id
        store = LocalStore(path=database, collection="memory", device_id=device_id)
        changelog = ChangeLog(database / "changes.sqlite3")
        try:
            records = store.all_points()
            if len(records) != 120:
                raise RuntimeError(f"expected 120 seed points for {device_id}, found {len(records)}")
            for record in records:
                changelog.append(record)
            queued += len(records)
        finally:
            changelog.close()
            store.close()
        print(f"queued {len(records)} initial snapshots for {device_id}", flush=True)
    return queued


def _spawn_services(
    services: list[tuple[str, subprocess.Popen]] | None = None,
) -> list[tuple[str, subprocess.Popen]]:
    if services is None:
        services = []

    cloud_environment = os.environ.copy()
    cloud_environment["EDGEMEM_CLOUD_DB"] = str(RUN_ROOT / "cloud.sqlite3")
    cloud_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "edgemem.cloud:app",
        "--host",
        BIND_HOST,
        "--port",
        str(SERVICE_PORTS["cloud"]),
        "--log-level",
        "warning",
    ]
    services.append(
        (
            "cloud",
            subprocess.Popen(
                cloud_command,
                cwd=ROOT,
                env=cloud_environment,
                start_new_session=True,
            ),
        )
    )
    _wait_ready("cloud", services[-1][1], CLOUD_URL)

    for device_id, url in DEVICE_URLS.items():
        environment = os.environ.copy()
        environment["EDGEMEM_RUN_DB"] = str(RUN_ROOT / device_id)
        environment["EDGEMEM_RUN_DEVICE_ID"] = device_id
        environment["EDGEMEM_CLOUD_URL"] = CLOUD_URL
        device_command = [
            sys.executable,
            "-m",
            "uvicorn",
            "scripts.run_all:create_device_app_from_environment",
            "--factory",
            "--host",
            BIND_HOST,
            "--port",
            str(SERVICE_PORTS[device_id]),
            "--log-level",
            "warning",
        ]
        services.append(
            (
                device_id,
                subprocess.Popen(
                    device_command,
                    cwd=ROOT,
                    env=environment,
                    start_new_session=True,
                ),
            )
        )
        _wait_ready(device_id, services[-1][1], url)
    return services


def _bootstrap_cloud() -> None:
    """Sync both initial snapshots and ensure every participant sees the merge."""
    for url in DEVICE_URLS.values():
        _http_json(url, "/network", method="POST", body={"online": True})
    for device_id, url in DEVICE_URLS.items():
        result = _http_json(url, "/sync-now", method="POST")
        if not isinstance(result, dict) or result.get("status") != "synced":
            raise RuntimeError(f"initial sync failed for {device_id}: {result}")
        print(
            f"bootstrapped {device_id}: pushed={result['pushed_changes']}, "
            f"pending={result['pending']}, cloud_seq={result['last_cloud_seq']}",
            flush=True,
        )
    for device_id, url in DEVICE_URLS.items():
        if device_id == "device_a":
            _http_json(url, "/sync-now", method="POST")
    for url in DEVICE_URLS.values():
        _http_json(url, "/network", method="POST", body={"online": False})

    states = []
    for url in DEVICE_URLS.values():
        points = _http_json(url, "/state")
        if not isinstance(points, list):
            raise RuntimeError(f"device {url} did not return a state list")
        states.append({point["id"]: point for point in points})
    cloud = _http_json(CLOUD_URL, "/state")
    if not isinstance(cloud, dict) or not isinstance(cloud.get("state"), dict):
        raise RuntimeError("cloud returned an invalid state response")

    def digest(state: dict[str, Any]) -> str:
        return json.dumps(state, sort_keys=True, separators=(",", ":"))

    expected = digest(cloud["state"])
    if digest(states[0]) != expected or digest(states[1]) != expected:
        raise RuntimeError("initial device/cloud states did not converge")
    print(
        f"seed state converged: {len(cloud['state'])} points, cloud_seq={cloud['cloud_seq']}",
        flush=True,
    )


def _shutdown(services: list[tuple[str, subprocess.Popen]]) -> None:
    for _, process in reversed(services):
        if process.poll() is None:
            process.terminate()
    for name, process in reversed(services):
        try:
            process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            print(f"force-stopping {name}", flush=True)
            process.kill()
            process.wait(timeout=3.0)


def main() -> int:
    """Prepare fresh seed stores, start services, and serve until Ctrl+C."""
    services: list[tuple[str, subprocess.Popen]] = []
    try:
        _preflight_ports()
        expected_root = (ROOT / "data" / "run").resolve()
        if expected_root != RUN_ROOT.resolve() or expected_root == (ROOT / "data").resolve():
            raise RuntimeError("refusing to reset an unexpected data/run path")
        if RUN_ROOT.exists():
            shutil.rmtree(RUN_ROOT)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        _ingest_devices()
        _queue_seed_snapshots()
        _spawn_services(services)
        _bootstrap_cloud()
        print("\nEdgeMem demo is ready:", flush=True)
        print(f"  Device A: {DEVICE_URLS['device_a']}", flush=True)
        print(f"  Device B: {DEVICE_URLS['device_b']}", flush=True)
        print(f"  Cloud:    {CLOUD_URL}", flush=True)
        print(f"  UI file:  {ROOT / 'edgemem' / 'ui' / 'index.html'}", flush=True)
        print("Open the UI file in a browser. Press Ctrl+C here to stop all services.", flush=True)

        while True:
            for name, process in services:
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(f"{name} exited unexpectedly with code {exit_code}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping EdgeMem demo services...", flush=True)
    except Exception as error:
        print(f"run_all failed: {error}", file=sys.stderr, flush=True)
        return_code = 1
        return return_code
    finally:
        _shutdown(services)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
