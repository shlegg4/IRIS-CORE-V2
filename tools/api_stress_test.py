#!/usr/bin/env python3
"""Concurrent stress test for a running IRIS REST API.

The default workload is read-only. Add --mutate to exercise the runtime command
queue with rapid synchronizer, preview, and shared-memory updates. This script
does not start or stop IRIS; run it against a disposable local instance when
using --mutate.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from http.client import HTTPConnection
from urllib.parse import urlsplit


@dataclass
class Stats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    total: int = 0
    transport_errors: int = 0
    http_errors: int = 0
    command_failures: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    first_failure: str | None = None
    failure_event: threading.Event = field(default_factory=threading.Event)

    def record(self, elapsed_ms: float, status: int | None, error: str | None, body: object, label: str) -> None:
        with self.lock:
            self.total += 1
            self.latencies_ms.append(elapsed_ms)
            if error:
                self.transport_errors += 1
                self.failure_event.set()
                if self.first_failure is None:
                    self.first_failure = f"{label}: {error}"
                if len(self.samples) < 10:
                    self.samples.append(f"{label}: {error}")
            elif status is not None and status >= 400:
                self.http_errors += 1
                if status >= 500:
                    self.failure_event.set()
                    if self.first_failure is None:
                        self.first_failure = f"{label}: HTTP {status}: {body}"
                if len(self.samples) < 10:
                    self.samples.append(f"{label}: HTTP {status}: {body}")
            elif isinstance(body, dict) and body.get("status") == "failed":
                self.command_failures += 1
                if len(self.samples) < 10:
                    self.samples.append(f"{label}: command failed: {body.get('message', '')}")
                self.failure_event.set()
                if self.first_failure is None:
                    self.first_failure = f"{label}: command failed: {body.get('message', '')}"


class Client:
    def __init__(self, base_url: str, timeout: float, stats: Stats) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("--base-url must be an http URL")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.prefix = parsed.path.rstrip("/")
        self.timeout = timeout
        self.stats = stats

    def request(self, method: str, path: str, payload: object | None = None, label: str | None = None) -> object | None:
        encoded = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"accept": "application/json"}
        if encoded is not None:
            headers["content-type"] = "application/json"
        started = time.perf_counter()
        status: int | None = None
        body: object = None
        error: str | None = None
        connection = HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request(method, f"{self.prefix}{path}", encoded, headers)
            response = connection.getresponse()
            status = response.status
            raw = response.read(2 * 1024 * 1024)
            body = json.loads(raw.decode("utf-8")) if raw else None
        except Exception as cause:  # ConnectionResetError, timeout, and refused connections matter here.
            error = f"{method} {path}: {type(cause).__name__}: {cause}"
        finally:
            connection.close()
            self.stats.record((time.perf_counter() - started) * 1000.0, status, error, body, label or f"{method} {path}")
        return body


def read_worker(client: Client, stop: threading.Event, seed: int) -> None:
    randomizer = random.Random(seed)
    endpoints = [
        ("GET", "/status"),
        ("GET", "/metrics"),
        ("GET", "/cameras"),
        ("GET", "/synchronizer"),
        ("GET", "/outputs/preview"),
        ("GET", "/calibration"),
    ]
    while not stop.is_set() and not client.stats.failure_event.is_set():
        method, path = randomizer.choice(endpoints)
        client.request(method, path)
        time.sleep(randomizer.uniform(0.005, 0.03))


def mutation_worker(client: Client, stop: threading.Event, seed: int) -> None:
    randomizer = random.Random(seed)
    index = 0
    while not stop.is_set() and not client.stats.failure_event.is_set():
        index += 1
        choice = index % 3
        if choice == 0:
            payload = {
                "tolerance_ms": 10 + (index % 4) * 5,
                "queue_capacity": 4 + (index % 2),
                "incomplete_batch_policy": "partial" if index % 2 else "drop",
            }
            client.request("PATCH", "/synchronizer", payload, "synchronizer update")
        elif choice == 1:
            client.request(
                "PATCH",
                "/outputs/preview",
                {
                    "http_enabled": True,
                    "mjpeg_enabled": True,
                    "h264_enabled": True,
                    "bind_address": "127.0.0.1",
                    "port": 8080,
                    "max_fps": 15 + index % 3,
                    "max_width": 1280,
                    "jpeg_quality": 80,
                    "bitrate": 2_000_000,
                    "queue_capacity": 4 + index % 2,
                },
                "preview update",
            )
        else:
            client.request(
                "PATCH",
                "/outputs/shared-memory",
                {"enabled": bool(index % 2), "destination": "iris-output", "capacity_bytes": 16 * 1024 * 1024, "legacy_v1": True},
                "shared-memory update",
            )
        time.sleep(randomizer.uniform(0.02, 0.08))


ROTATIONS = ("none", "cw90", "180", "ccw90")


def camera_payload(camera: dict[str, object], rotation: str | None = None) -> dict[str, object]:
    """Convert GET /cameras output into the POST /cameras shape."""
    payload: dict[str, object] = {"camera_id": camera["camera_id"]}
    fields = (
        "device_symbolic_link", "device_index", "width", "height", "format", "cuda_device",
        "sample_queue_capacity", "frame_pool_capacity", "overflow", "allow_format_fallback", "reconnect",
    )
    for field_name in fields:
        value = camera.get(field_name)
        if value is not None:
            payload[field_name] = value
    frame_rate = camera.get("frame_rate")
    if isinstance(frame_rate, dict):
        payload["frame_rate"] = {"numerator": frame_rate.get("numerator", 30), "denominator": frame_rate.get("denominator", 1)}
    elif frame_rate is not None:
        payload["frame_rate"] = frame_rate
    payload["rotation"] = rotation or str(camera.get("rotation", "none"))
    return payload


def camera_churn_worker(client: Client, stop: threading.Event, cameras: list[dict[str, object]], seed: int, mode: str) -> None:
    randomizer = random.Random(seed)
    index = 0
    while not stop.is_set() and not client.stats.failure_event.is_set():
        camera = randomizer.choice(cameras)
        camera_id = camera["camera_id"]
        index += 1
        rotation = ROTATIONS[index % len(ROTATIONS)]
        if mode in ("rotate", "both"):
            client.request("PATCH", f"/cameras/{camera_id}", {"rotation": rotation}, f"camera {camera_id} rotation={rotation}")
        # Removal is intentionally paired with re-addition, so the test keeps
        # exercising the full lifecycle rather than permanently changing the rig.
        if len(cameras) > 1 and (mode == "remove" or index % 3 == 0) and not client.stats.failure_event.is_set():
            client.request("DELETE", f"/cameras/{camera_id}", label=f"camera {camera_id} remove")
            client.request("POST", "/cameras", camera_payload(camera, rotation), f"camera {camera_id} re-add")
        time.sleep(randomizer.uniform(0.005, 0.025))


def restore_cameras(client: Client, original: list[dict[str, object]]) -> None:
    """Best-effort restoration after a churn run."""
    if not original:
        return
    current = client.request("GET", "/cameras", label="restore list")
    if not isinstance(current, list):
        return
    current_by_id = {item.get("camera_id"): item for item in current if isinstance(item, dict)}
    original_ids = {item.get("camera_id") for item in original}
    for camera_id, item in list(current_by_id.items()):
        if camera_id not in original_ids and len(current_by_id) > 1:
            client.request("DELETE", f"/cameras/{camera_id}", label=f"restore remove {camera_id}")
            current_by_id.pop(camera_id, None)
    for camera in original:
        camera_id = camera.get("camera_id")
        payload = camera_payload(camera)
        if camera_id not in current_by_id:
            client.request("POST", "/cameras", payload, f"restore add {camera_id}")
        else:
            patch = {key: value for key, value in payload.items() if key != "camera_id"}
            client.request("PATCH", f"/cameras/{camera_id}", patch, f"restore camera {camera_id}")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/api/v1")
    parser.add_argument("--duration", type=float, default=30.0, help="duration in seconds (default: 30)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent read workers (default: 8)")
    parser.add_argument("--timeout", type=float, default=2.0, help="per-request timeout in seconds")
    parser.add_argument("--mutate", action="store_true", help="also issue rapid runtime configuration updates")
    parser.add_argument("--camera-churn", action="store_true", help="rapidly rotate, remove, and re-add configured cameras")
    parser.add_argument("--camera-workers", type=int, default=2, help="concurrent camera lifecycle workers (default: 2)")
    parser.add_argument("--seed-camera", action="store_true", help="when only one camera exists, add a temporary clone so removal can be exercised")
    parser.add_argument("--camera-mode", choices=("both", "rotate", "remove"), default="both", help="camera operation to stress (default: both)")
    args = parser.parse_args()
    if args.duration <= 0 or args.workers <= 0 or args.timeout <= 0 or args.camera_workers <= 0:
        parser.error("duration, workers, timeout, and camera-workers must be positive")

    stats = Stats()
    client = Client(args.base_url, args.timeout, stats)
    print(f"Checking {args.base_url} ...", flush=True)
    if client.request("GET", "/status", label="initial status") is None and stats.transport_errors:
        print("IRIS API was not reachable; aborting.", file=sys.stderr)
        return 2

    original_cameras: list[dict[str, object]] = []
    churn_cameras: list[dict[str, object]] = []
    if args.camera_churn:
        camera_response = client.request("GET", "/cameras", label="initial cameras")
        if isinstance(camera_response, list):
            original_cameras = [item for item in camera_response if isinstance(item, dict)]
        churn_cameras = list(original_cameras)
        if not original_cameras:
            print("No configured cameras found; camera churn will not run.", file=sys.stderr)
        elif args.seed_camera and len(original_cameras) == 1:
            source = original_cameras[0]
            temporary = dict(source)
            temporary["camera_id"] = max(int(source.get("camera_id", 0)), 0) + 1
            seeded = client.request("POST", "/cameras", camera_payload(temporary), "seed temporary camera")
            if isinstance(seeded, dict) and seeded.get("status") == "applied":
                churn_cameras.append(temporary)
                print(f"Seeded temporary camera {temporary['camera_id']} for removal testing.", flush=True)
            else:
                print("Could not seed a second camera; removal testing is disabled.", file=sys.stderr)

    stop = threading.Event()
    threads = [threading.Thread(target=read_worker, args=(client, stop, i), daemon=True) for i in range(args.workers)]
    if args.mutate:
        threads.append(threading.Thread(target=mutation_worker, args=(client, stop, args.workers + 1), daemon=True))
    if args.camera_churn and churn_cameras:
        for index in range(args.camera_workers):
            threads.append(threading.Thread(target=camera_churn_worker, args=(client, stop, churn_cameras, args.workers + index + 10, args.camera_mode), daemon=True))
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + args.duration
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or stats.failure_event.is_set():
                break
            time.sleep(min(1.0, remaining))
            if time.monotonic() >= deadline:
                break
            with stats.lock:
                print(f"requests={stats.total} transport_errors={stats.transport_errors} http_errors={stats.http_errors}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; collecting results.", flush=True)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=args.timeout + 1.0)
        if args.camera_churn and original_cameras and not stats.transport_errors:
            restore_cameras(client, original_cameras)

    with stats.lock:
        latencies = list(stats.latencies_ms)
        print("\nAPI stress result")
        print(f"requests: {stats.total}")
        print(f"transport errors: {stats.transport_errors}")
        print(f"HTTP errors: {stats.http_errors}")
        print(f"command failures: {stats.command_failures}")
        print(f"latency ms: p50={percentile(latencies, .50):.1f} p95={percentile(latencies, .95):.1f} max={max(latencies, default=0):.1f}")
        if stats.first_failure:
            print(f"first fatal failure: {stats.first_failure}")
        if stats.samples:
            print("samples:")
            for sample in stats.samples:
                print(f"  {sample}")
        return 1 if stats.failure_event.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
