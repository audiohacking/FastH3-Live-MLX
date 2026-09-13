"""Live dashboard helpers: log ring, host metrics, HTML page."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from h3_backend import apple_chip_brand, physical_memory_bytes

DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")

_log_lock = threading.Lock()
_log_lines: deque[dict] = deque(maxlen=800)
_log_id = 0


class RingLogHandler(logging.Handler):
    """Capture live logger lines for the browser console."""

    def emit(self, record: logging.LogRecord) -> None:
        global _log_id
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        with _log_lock:
            _log_id += 1
            _log_lines.append(
                {
                    "id": _log_id,
                    "t": time.strftime("%H:%M:%S", time.localtime(record.created)),
                    "level": record.levelno,
                    "msg": msg,
                }
            )


def attach_ring_logger(logger_name: str = "h3-live") -> None:
    root = logging.getLogger(logger_name)
    if any(isinstance(h, RingLogHandler) for h in root.handlers):
        return
    handler = RingLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)


def logs_after(after_id: int = 0) -> list[dict]:
    with _log_lock:
        return [line for line in _log_lines if line["id"] > after_id]


def _vm_used_bytes() -> int | None:
    if not Path("/usr/bin/vm_stat").exists():
        return None
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    page = 16384
    vals: dict[str, int] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        raw = raw.strip().rstrip(".")
        if raw.isdigit():
            vals[key.strip()] = int(raw)
        if "page size of" in line:
            try:
                page = int(line.split("page size of")[1].split()[0])
            except (IndexError, ValueError):
                pass
    # Approx used = total - free - speculative (inactive still "used")
    total = physical_memory_bytes()
    if total is None:
        return None
    free = (vals.get("Pages free", 0) + vals.get("Pages speculative", 0)) * page
    return max(0, total - free)


def _h3_proc_stats() -> tuple[float | None, float | None]:
    """Return (rss_gb, cpu_pct) for the live ./h3 child, if any."""
    try:
        out = subprocess.check_output(
            ["ps", "-axo", "pid=,pcpu=,rss=,command="], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    best = None
    for line in out.splitlines():
        if "/third_party/h3.c/h3" not in line and not line.rstrip().endswith("/h3"):
            continue
        if "liveserver" in line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        try:
            cpu = float(parts[1])
            rss_kb = float(parts[2])
        except ValueError:
            continue
        rss_gb = rss_kb / (1024 * 1024)
        if best is None or rss_gb > best[0]:
            best = (rss_gb, cpu)
    if best is None:
        return None, None
    return best


def _system_cpu_pct() -> float | None:
    """Host-wide CPU busy % via ``ps`` (sum of process %cpu, capped at ncpu*100)."""
    try:
        out = subprocess.check_output(["ps", "-A", "-o", "%cpu="], text=True)
        ncpu = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    total = 0.0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total += float(line)
        except ValueError:
            continue
    if ncpu < 1:
        return None
    # Normalize to 0–100 of the machine (ps reports per-core %).
    return min(100.0, total / ncpu)


def _gpu_util_pct() -> float | None:
    """Apple GPU Device Utilization % from AGXAccelerator (no sudo)."""
    try:
        raw = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-a", "-c", "AGXAccelerator"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        import plistlib

        entries = plistlib.loads(raw)
    except Exception:
        return None
    if not isinstance(entries, list):
        return None
    best = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stats = entry.get("PerformanceStatistics") or {}
        if not isinstance(stats, dict):
            continue
        val = stats.get("Device Utilization %")
        if isinstance(val, (int, float)):
            best = float(val) if best is None else max(best, float(val))
    return best


def host_metrics() -> dict:
    total = physical_memory_bytes()
    used = _vm_used_bytes()
    rss, h3_cpu = _h3_proc_stats()
    return {
        "chip": apple_chip_brand() or None,
        "ram_total_gb": None if total is None else total / (1024**3),
        "ram_used_gb": None if used is None else used / (1024**3),
        "h3_rss_gb": rss,
        "h3_cpu_pct": h3_cpu,
        "cpu_pct": _system_cpu_pct(),
        "gpu_pct": _gpu_util_pct(),
    }


def load_dashboard_html() -> bytes:
    return DASHBOARD_HTML.read_bytes()
