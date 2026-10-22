"""Google Cluster Workload Trace Dataset Adapter for BACCP.

Provides ingestion, schema validation, normalization, and characteristic extraction
for public Google Cluster Workload Traces (e.g., clusterdata-2011-2 and clusterdata-2019a).

Dataset Source:
- Official Google Cluster Data Repository: https://github.com/google/cluster-data
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- GCS Bucket: gs://clusterdata-2011-2/

Mappings to Airline IT Workload Tiers:
- Scheduling Class 3 / Priority 9-11 -> Flight Reservations (Latency-sensitive production)
- Scheduling Class 2 / Priority 6-8  -> Crew Operations (Operational scheduling)
- Scheduling Class 0-1 / Priority 0-5 -> Baggage Reconciliation & Batch Analytics (Batch shed-tolerant)
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import datetime
import gzip
import io
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger("baccp.workload.google_cluster")

DATASET_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = DATASET_ROOT / "sample_data"
DEFAULT_TASK_EVENTS_SAMPLE = SAMPLE_DATA_DIR / "google_cluster_task_events_sample.csv"
DEFAULT_TASK_USAGE_SAMPLE = SAMPLE_DATA_DIR / "google_cluster_task_usage_sample.csv"
DEFAULT_JSON_SAMPLE = SAMPLE_DATA_DIR / "google_cluster_sample.json"

# Event type mapping defined by Google clusterdata-2011-2 specification
EVENT_TYPES = {
    0: "SUBMIT",
    1: "SCHEDULE",
    2: "EVICT",
    3: "FAIL",
    4: "FINISH",
    5: "KILL",
    6: "LOST",
    7: "UPDATE_PENDING",
    8: "UPDATE_RUNNING",
}

# Airline IT domain tier classification derived from scheduling class and priority
def map_to_airline_tier(scheduling_class: int, priority: int) -> str:
    """Map Google Cluster scheduling class and priority to airline microservice tier."""
    if scheduling_class >= 3 or priority >= 9:
        return "production_reservations"
    elif scheduling_class == 2 or priority >= 6:
        return "mid_crew"
    else:
        return "batch_baggage"


@dataclass
class GoogleClusterTaskEvent:
    """Represents a single lifecycle event from Google Cluster Data task_events table."""

    timestamp_usec: int
    timestamp_sec: float
    missing_info: bool
    job_id: int
    task_index: int
    machine_id: Optional[str]
    event_type: int
    event_name: str
    user_name: Optional[str]
    scheduling_class: int
    priority: int
    cpu_request: float
    memory_request: float
    disk_space_request: float
    different_machine_restriction: bool
    airline_tier: str
    raw_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return asdict(self)


@dataclass
class GoogleClusterTaskUsage:
    """Represents a 5-minute resource utilization window from task_usage table."""

    start_time_usec: int
    end_time_usec: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    job_id: int
    task_index: int
    machine_id: Optional[str]
    cpu_usage: float
    memory_usage: float
    assigned_memory_usage: float
    unmapped_page_cache_memory_usage: float
    total_page_cache_memory_usage: float
    disk_io_time: float
    local_disk_space_usage: float
    max_cpu_usage: float
    max_memory_usage: float
    max_disk_io_time: float
    cpu_rate_1s_sample: Optional[float]
    raw_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize usage record to dictionary."""
        return asdict(self)


class GoogleClusterTraceAdapter:
    """Dataset adapter for parsing, normalizing, and aggregating Google Cluster Data traces."""

    def __init__(
        self,
        task_events_path: Optional[Union[str, Path]] = None,
        task_usage_path: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
    ):
        self.task_events_path = Path(task_events_path) if task_events_path else DEFAULT_TASK_EVENTS_SAMPLE
        self.task_usage_path = Path(task_usage_path) if task_usage_path else DEFAULT_TASK_USAGE_SAMPLE
        self.json_path = Path(json_path) if json_path else DEFAULT_JSON_SAMPLE
        self.strict = strict

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Return dataset provenance, schema metadata, and licensing."""
        return {
            "name": "Google Cluster Data (clusterdata-2011-2 / clusterdata-2019a)",
            "official_url": "https://github.com/google/cluster-data",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "gcs_bucket": "gs://clusterdata-2011-2/",
            "primary_tables": ["task_events", "task_usage", "job_events", "machine_events"],
            "temporal_resolution": {
                "task_events": "microsecond-level lifecycle transitions",
                "task_usage": "300-second (5-minute) aggregate sampling windows",
            },
            "normalization": "Resource request and usage metrics normalized to [0, 1] relative to machine capacity.",
            "airline_mapping": {
                "class_3_priority_9_11": "Flight Reservations (critical, latency-sensitive)",
                "class_2_priority_6_8": "Crew Operations (operational scheduling)",
                "class_0_1_priority_0_5": "Baggage Handling & Reporting (batch, shed-tolerant)",
            },
        }

    @staticmethod
    def download_instructions() -> str:
        """Return explicit instructions for acquiring the full public dataset."""
        return (
            "To download the full public Google Cluster Data:\n"
            "1. Install the Google Cloud SDK (gsutil).\n"
            "2. Access the public bucket via:\n"
            "   gsutil -m cp gs://clusterdata-2011-2/task_events/part-00000-of-00500.csv.gz ./data/\n"
            "   gsutil -m cp gs://clusterdata-2011-2/task_usage/part-00000-of-00500.csv.gz ./data/\n"
            "3. Alternatively, for the 2019 Borg traces, query Google Cloud BigQuery public dataset:\n"
            "   `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`\n"
            "Note: The full dataset is ~40GB compressed / ~2TB uncompressed. Do not download automatically."
        )

    def _open_file(self, path: Path) -> io.TextIOBase:
        """Open plain or gzip-compressed file in text mode."""
        if not path.is_file():
            raise FileNotFoundError(f"Google Cluster Data file not found at: {path}")
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return open(path, "r", encoding="utf-8", errors="replace")

    def _clean_float(self, value: Any, default: float = 0.0) -> float:
        """Safely parse and clean float values, clamping between 0.0 and positive infinity."""
        if value is None or value == "":
            return default
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return default
            return max(0.0, val)
        except (ValueError, TypeError):
            if self.strict:
                raise ValueError(f"Invalid float value: {value!r}")
            return default

    def _clean_int(self, value: Any, default: int = 0) -> int:
        """Safely parse integer value."""
        if value is None or value == "":
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            if self.strict:
                raise ValueError(f"Invalid integer value: {value!r}")
            return default

    def parse_task_event_row(self, row: List[str], header: Optional[List[str]] = None) -> Optional[GoogleClusterTaskEvent]:
        """Parse a single task_events row (supports headerless standard CSV and header-based)."""
        if not row or len(row) < 6:
            if self.strict:
                raise ValueError(f"Insufficient columns in task_events row: {row}")
            return None

        # Ignore comment lines
        if row[0].startswith("#"):
            return None

        raw = {}
        if header and len(header) == len(row):
            raw = dict(zip(header, row))
            ts_val = raw.get("timestamp") or raw.get("timestamp_usec") or row[0]
            missing_val = raw.get("missing_info", row[1] if len(row) > 1 else 0)
            job_val = raw.get("job_id", row[2] if len(row) > 2 else 0)
            task_val = raw.get("task_index", row[3] if len(row) > 3 else 0)
            machine_val = raw.get("machine_id", row[4] if len(row) > 4 else None)
            event_val = raw.get("event_type", row[5] if len(row) > 5 else 0)
            user_val = raw.get("user_name", row[6] if len(row) > 6 else "")
            sched_val = raw.get("scheduling_class", row[7] if len(row) > 7 else 0)
            prio_val = raw.get("priority", row[8] if len(row) > 8 else 0)
            cpu_val = raw.get("cpu_request", row[9] if len(row) > 9 else 0.0)
            mem_val = raw.get("memory_request", row[10] if len(row) > 10 else 0.0)
            disk_val = raw.get("disk_space_request", row[11] if len(row) > 11 else 0.0)
            diff_val = raw.get("different_machine_restriction", row[12] if len(row) > 12 else 0)
        else:
            # Headerless official Google Cluster trace column indices:
            # 0:timestamp, 1:missing_info, 2:job_id, 3:task_index, 4:machine_id, 5:event_type,
            # 6:user_name, 7:scheduling_class, 8:priority, 9:cpu_request, 10:memory_request,
            # 11:disk_space_request, 12:different_machine_restriction
            raw = {f"col_{i}": v for i, v in enumerate(row)}
            ts_val = row[0]
            missing_val = row[1] if len(row) > 1 else 0
            job_val = row[2] if len(row) > 2 else 0
            task_val = row[3] if len(row) > 3 else 0
            machine_val = row[4] if len(row) > 4 else None
            event_val = row[5] if len(row) > 5 else 0
            user_val = row[6] if len(row) > 6 else ""
            sched_val = row[7] if len(row) > 7 else 0
            prio_val = row[8] if len(row) > 8 else 0
            cpu_val = row[9] if len(row) > 9 else 0.0
            mem_val = row[10] if len(row) > 10 else 0.0
            disk_val = row[11] if len(row) > 11 else 0.0
            diff_val = row[12] if len(row) > 12 else 0

        ts_usec = self._clean_int(ts_val, 0)
        event_code = self._clean_int(event_val, 0)
        sched_class = self._clean_int(sched_val, 0)
        priority = self._clean_int(prio_val, 0)

        # Impute default CPU/memory requests if missing_info is set
        cpu_req = self._clean_float(cpu_val, default=0.05 if sched_class >= 2 else 0.02)
        mem_req = self._clean_float(mem_val, default=0.04 if sched_class >= 2 else 0.02)
        disk_req = self._clean_float(disk_val, default=0.001)

        machine_id_clean = str(machine_val).strip() if machine_val and str(machine_val).strip() else None

        return GoogleClusterTaskEvent(
            timestamp_usec=ts_usec,
            timestamp_sec=round(ts_usec / 1_000_000.0, 6),
            missing_info=bool(self._clean_int(missing_val, 0)),
            job_id=self._clean_int(job_val, 0),
            task_index=self._clean_int(task_val, 0),
            machine_id=machine_id_clean,
            event_type=event_code,
            event_name=EVENT_TYPES.get(event_code, f"UNKNOWN_{event_code}"),
            user_name=str(user_val).strip() if user_val else None,
            scheduling_class=sched_class,
            priority=priority,
            cpu_request=cpu_req,
            memory_request=mem_req,
            disk_space_request=disk_req,
            different_machine_restriction=bool(self._clean_int(diff_val, 0)),
            airline_tier=map_to_airline_tier(sched_class, priority),
            raw_fields=raw,
        )

    def parse_task_usage_row(self, row: List[str], header: Optional[List[str]] = None) -> Optional[GoogleClusterTaskUsage]:
        """Parse a single task_usage row (supports headerless standard CSV and header-based)."""
        if not row or len(row) < 7:
            if self.strict:
                raise ValueError(f"Insufficient columns in task_usage row: {row}")
            return None

        # Ignore comments
        if row[0].startswith("#"):
            return None

        raw = {}
        if header and len(header) == len(row):
            raw = dict(zip(header, row))
            start_val = raw.get("start_time", row[0])
            end_val = raw.get("end_time", row[1] if len(row) > 1 else 0)
            job_val = raw.get("job_id", row[2] if len(row) > 2 else 0)
            task_val = raw.get("task_index", row[3] if len(row) > 3 else 0)
            machine_val = raw.get("machine_id", row[4] if len(row) > 4 else None)
            cpu_val = raw.get("cpu_usage", row[5] if len(row) > 5 else 0.0)
            mem_val = raw.get("memory_usage", row[6] if len(row) > 6 else 0.0)
            assigned_val = raw.get("assigned_memory_usage", row[7] if len(row) > 7 else mem_val)
            unmapped_val = raw.get("unmapped_page_cache_memory_usage", row[8] if len(row) > 8 else 0.0)
            cache_val = raw.get("total_page_cache_memory_usage", row[9] if len(row) > 9 else 0.0)
            io_val = raw.get("disk_io_time", row[10] if len(row) > 10 else 0.0)
            disk_val = raw.get("local_disk_space_usage", row[11] if len(row) > 11 else 0.0)
            max_cpu_val = raw.get("max_cpu_usage", row[12] if len(row) > 12 else cpu_val)
            max_mem_val = raw.get("max_memory_usage", row[13] if len(row) > 13 else mem_val)
            max_io_val = raw.get("max_disk_io_time", row[14] if len(row) > 14 else io_val)
            sample_cpu_val = raw.get("cpu_rate_1s_sample", row[15] if len(row) > 15 else None)
        else:
            raw = {f"col_{i}": v for i, v in enumerate(row)}
            start_val = row[0]
            end_val = row[1] if len(row) > 1 else 0
            job_val = row[2] if len(row) > 2 else 0
            task_val = row[3] if len(row) > 3 else 0
            machine_val = row[4] if len(row) > 4 else None
            cpu_val = row[5] if len(row) > 5 else 0.0
            mem_val = row[6] if len(row) > 6 else 0.0
            assigned_val = row[7] if len(row) > 7 else mem_val
            unmapped_val = row[8] if len(row) > 8 else 0.0
            cache_val = row[9] if len(row) > 9 else 0.0
            io_val = row[10] if len(row) > 10 else 0.0
            disk_val = row[11] if len(row) > 11 else 0.0
            max_cpu_val = row[12] if len(row) > 12 else cpu_val
            max_mem_val = row[13] if len(row) > 13 else mem_val
            max_io_val = row[14] if len(row) > 14 else io_val
            sample_cpu_val = row[15] if len(row) > 15 else None

        start_usec = self._clean_int(start_val, 0)
        end_usec = self._clean_int(end_val, start_usec + 300_000_000)
        start_sec = round(start_usec / 1_000_000.0, 6)
        end_sec = round(end_usec / 1_000_000.0, 6)
        duration_sec = max(0.0, end_sec - start_sec)

        cpu_usage = self._clean_float(cpu_val, 0.0)
        mem_usage = self._clean_float(mem_val, 0.0)
        max_cpu = max(cpu_usage, self._clean_float(max_cpu_val, cpu_usage))
        max_mem = max(mem_usage, self._clean_float(max_mem_val, mem_usage))

        sample_cpu = self._clean_float(sample_cpu_val) if sample_cpu_val is not None and str(sample_cpu_val).strip() else None
        machine_clean = str(machine_val).strip() if machine_val and str(machine_val).strip() else None

        return GoogleClusterTaskUsage(
            start_time_usec=start_usec,
            end_time_usec=end_usec,
            start_time_sec=start_sec,
            end_time_sec=end_sec,
            duration_sec=duration_sec,
            job_id=self._clean_int(job_val, 0),
            task_index=self._clean_int(task_val, 0),
            machine_id=machine_clean,
            cpu_usage=cpu_usage,
            memory_usage=mem_usage,
            assigned_memory_usage=self._clean_float(assigned_val, mem_usage),
            unmapped_page_cache_memory_usage=self._clean_float(unmapped_val, 0.0),
            total_page_cache_memory_usage=self._clean_float(cache_val, 0.0),
            disk_io_time=self._clean_float(io_val, 0.0),
            local_disk_space_usage=self._clean_float(disk_val, 0.0),
            max_cpu_usage=max_cpu,
            max_memory_usage=max_mem,
            max_disk_io_time=self._clean_float(max_io_val, 0.0),
            cpu_rate_1s_sample=sample_cpu,
            raw_fields=raw,
        )

    def load_task_events(
        self,
        path: Optional[Union[str, Path]] = None,
        limit: Optional[int] = None,
    ) -> List[GoogleClusterTaskEvent]:
        """Load and parse task_events from a CSV or gzipped CSV file."""
        target_path = Path(path) if path else self.task_events_path
        events: List[GoogleClusterTaskEvent] = []

        with self._open_file(target_path) as f:
            reader = csv.reader(f)
            header = None
            first_row = True

            for row in reader:
                if not row:
                    continue
                # Detect header row
                if first_row and any("timestamp" in col.lower() or "job" in col.lower() for col in row):
                    header = [c.strip().lower() for c in row]
                    first_row = False
                    continue
                first_row = False

                event = self.parse_task_event_row(row, header)
                if event is not None:
                    events.append(event)
                    if limit and len(events) >= limit:
                        break

        logger.info(f"Loaded {len(events)} Google Cluster task events from {target_path}")
        return events

    def load_task_usage(
        self,
        path: Optional[Union[str, Path]] = None,
        limit: Optional[int] = None,
    ) -> List[GoogleClusterTaskUsage]:
        """Load and parse task_usage from a CSV or gzipped CSV file."""
        target_path = Path(path) if path else self.task_usage_path
        usages: List[GoogleClusterTaskUsage] = []

        with self._open_file(target_path) as f:
            reader = csv.reader(f)
            header = None
            first_row = True

            for row in reader:
                if not row:
                    continue
                # Detect header
                if first_row and any("time" in col.lower() or "usage" in col.lower() for col in row):
                    header = [c.strip().lower() for c in row]
                    first_row = False
                    continue
                first_row = False

                usage = self.parse_task_usage_row(row, header)
                if usage is not None:
                    usages.append(usage)
                    if limit and len(usages) >= limit:
                        break

        logger.info(f"Loaded {len(usages)} Google Cluster task usage records from {target_path}")
        return usages

    def load_from_json(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[GoogleClusterTaskEvent], List[GoogleClusterTaskUsage]]:
        """Load task events and task usage from a consolidated JSON document."""
        target_path = Path(path) if path else self.json_path
        if not target_path.is_file():
            raise FileNotFoundError(f"Google Cluster JSON sample not found at: {target_path}")

        data = json.loads(target_path.read_text(encoding="utf-8"))
        events: List[GoogleClusterTaskEvent] = []
        usages: List[GoogleClusterTaskUsage] = []

        for e_dict in data.get("task_events", []):
            ts_usec = self._clean_int(e_dict.get("timestamp_usec") or e_dict.get("timestamp", 0))
            event_type = self._clean_int(e_dict.get("event_type", 0))
            sched = self._clean_int(e_dict.get("scheduling_class", 0))
            prio = self._clean_int(e_dict.get("priority", 0))

            event = GoogleClusterTaskEvent(
                timestamp_usec=ts_usec,
                timestamp_sec=round(ts_usec / 1_000_000.0, 6),
                missing_info=bool(e_dict.get("missing_info", False)),
                job_id=self._clean_int(e_dict.get("job_id", 0)),
                task_index=self._clean_int(e_dict.get("task_index", 0)),
                machine_id=str(e_dict.get("machine_id")) if e_dict.get("machine_id") else None,
                event_type=event_type,
                event_name=EVENT_TYPES.get(event_type, f"EVENT_{event_type}"),
                user_name=e_dict.get("user_name"),
                scheduling_class=sched,
                priority=prio,
                cpu_request=self._clean_float(e_dict.get("cpu_request", 0.05)),
                memory_request=self._clean_float(e_dict.get("memory_request", 0.04)),
                disk_space_request=self._clean_float(e_dict.get("disk_space_request", 0.001)),
                different_machine_restriction=bool(e_dict.get("different_machine_restriction", False)),
                airline_tier=map_to_airline_tier(sched, prio),
                raw_fields=e_dict,
            )
            events.append(event)

        for u_dict in data.get("task_usage", []):
            start_usec = self._clean_int(u_dict.get("start_time_usec") or u_dict.get("start_time", 0))
            end_usec = self._clean_int(u_dict.get("end_time_usec") or u_dict.get("end_time", start_usec + 300_000_000))
            start_sec = round(start_usec / 1_000_000.0, 6)
            end_sec = round(end_usec / 1_000_000.0, 6)

            usage = GoogleClusterTaskUsage(
                start_time_usec=start_usec,
                end_time_usec=end_usec,
                start_time_sec=start_sec,
                end_time_sec=end_sec,
                duration_sec=max(0.0, end_sec - start_sec),
                job_id=self._clean_int(u_dict.get("job_id", 0)),
                task_index=self._clean_int(u_dict.get("task_index", 0)),
                machine_id=str(u_dict.get("machine_id")) if u_dict.get("machine_id") else None,
                cpu_usage=self._clean_float(u_dict.get("cpu_usage", 0.0)),
                memory_usage=self._clean_float(u_dict.get("memory_usage", 0.0)),
                assigned_memory_usage=self._clean_float(u_dict.get("assigned_memory_usage", 0.0)),
                unmapped_page_cache_memory_usage=self._clean_float(u_dict.get("unmapped_page_cache_memory_usage", 0.0)),
                total_page_cache_memory_usage=self._clean_float(u_dict.get("total_page_cache_memory_usage", 0.0)),
                disk_io_time=self._clean_float(u_dict.get("disk_io_time", 0.0)),
                local_disk_space_usage=self._clean_float(u_dict.get("local_disk_space_usage", 0.0)),
                max_cpu_usage=self._clean_float(u_dict.get("max_cpu_usage", 0.0)),
                max_memory_usage=self._clean_float(u_dict.get("max_memory_usage", 0.0)),
                max_disk_io_time=self._clean_float(u_dict.get("max_disk_io_time", 0.0)),
                cpu_rate_1s_sample=self._clean_float(u_dict.get("cpu_rate_1s_sample")) if u_dict.get("cpu_rate_1s_sample") is not None else None,
                raw_fields=u_dict,
            )
            usages.append(usage)

        return events, usages

    def compute_workload_intensity_timeseries(
        self,
        window_seconds: float = 60.0,
        events: Optional[List[GoogleClusterTaskEvent]] = None,
        usage: Optional[List[GoogleClusterTaskUsage]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute aggregated workload intensity metrics across discrete temporal buckets."""
        if events is None:
            events = self.load_task_events()
        if usage is None:
            usage = self.load_task_usage()

        if not events and not usage:
            return []

        # Find temporal span
        min_sec = float("inf")
        max_sec = 0.0

        for e in events:
            min_sec = min(min_sec, e.timestamp_sec)
            max_sec = max(max_sec, e.timestamp_sec)

        for u in usage:
            min_sec = min(min_sec, u.start_time_sec)
            max_sec = max(max_sec, u.end_time_sec)

        if min_sec == float("inf"):
            min_sec = 0.0
        if max_sec <= min_sec:
            max_sec = min_sec + window_seconds

        num_buckets = max(1, math.ceil((max_sec - min_sec) / window_seconds))
        buckets: List[Dict[str, Any]] = []

        for b in range(num_buckets):
            b_start = min_sec + (b * window_seconds)
            b_end = b_start + window_seconds

            # Filter events falling into this bucket
            b_events = [e for e in events if b_start <= e.timestamp_sec < b_end]
            # Filter usage falling into or overlapping this bucket
            b_usage = [u for u in usage if (u.start_time_sec < b_end and u.end_time_sec > b_start)]

            submits = sum(1 for e in b_events if e.event_type == 0)
            schedules = sum(1 for e in b_events if e.event_type == 1)
            evictions = sum(1 for e in b_events if e.event_type == 2)
            failures = sum(1 for e in b_events if e.event_type == 3)
            finishes = sum(1 for e in b_events if e.event_type == 4)

            # Compute tier breakdown for submitted tasks
            tier_counts = {
                "production_reservations": sum(1 for e in b_events if e.airline_tier == "production_reservations"),
                "mid_crew": sum(1 for e in b_events if e.airline_tier == "mid_crew"),
                "batch_baggage": sum(1 for e in b_events if e.airline_tier == "batch_baggage"),
            }

            cpu_values = [u.cpu_usage for u in b_usage]
            mem_values = [u.memory_usage for u in b_usage]
            cpu_requests = [e.cpu_request for e in b_events]

            mean_cpu = float(sum(cpu_values) / len(cpu_values)) if cpu_values else 0.0
            peak_cpu = float(max(cpu_values)) if cpu_values else 0.0
            total_cpu_demand = float(sum(cpu_requests)) + float(sum(cpu_values))

            mean_mem = float(sum(mem_values) / len(mem_values)) if mem_values else 0.0
            peak_mem = float(max(mem_values)) if mem_values else 0.0

            arrival_rate = round(len(b_events) / window_seconds, 4)

            buckets.append({
                "window_index": b,
                "window_start_sec": round(b_start, 2),
                "window_end_sec": round(b_end, 2),
                "arrival_rate_tasks_per_sec": arrival_rate,
                "task_event_counts": {
                    "total": len(b_events),
                    "submit": submits,
                    "schedule": schedules,
                    "evict": evictions,
                    "fail": failures,
                    "finish": finishes,
                },
                "resource_metrics": {
                    "mean_cpu_usage": round(mean_cpu, 4),
                    "peak_cpu_usage": round(peak_cpu, 4),
                    "total_cpu_demand": round(total_cpu_demand, 4),
                    "mean_memory_usage": round(mean_mem, 4),
                    "peak_memory_usage": round(peak_mem, 4),
                },
                "airline_tier_distribution": tier_counts,
                "active_task_estimate": len(b_usage) + schedules - finishes - evictions - failures,
            })

        return buckets

    def extract_airline_workload_profile(
        self,
        window_seconds: float = 60.0,
    ) -> Dict[str, Any]:
        """Convert Google Cluster workload patterns into airline IT service telemetry rates.
        
        Transforms real cluster load dynamics into normalized RED metrics (Request rate,
        Error rate, Duration/latency) suitable for the BACCP orchestrator and cascade predictor.
        """
        series = self.compute_workload_intensity_timeseries(window_seconds=window_seconds)
        if not series:
            # Safe nominal defaults matching repository contracts
            return {
                "reservations": {"request_rate": 24.5, "latency_ms": 18.5, "error_rate": 0.0, "cpu_load": 0.25},
                "crew": {"request_rate": 14.8, "latency_ms": 22.1, "error_rate": 0.0, "cpu_load": 0.20},
                "baggage": {"request_rate": 18.2, "latency_ms": 15.4, "error_rate": 0.0, "cpu_load": 0.15},
                "boundary-gateway": {"request_rate": 57.5, "latency_ms": 45.0, "error_rate": 0.0, "cpu_load": 0.35},
                "legacy-core": {"request_rate": 57.5, "latency_ms": 68.2, "error_rate": 0.0, "cpu_load": 0.40},
            }

        first_window = series[0]
        res_count = first_window["airline_tier_distribution"]["production_reservations"]
        crew_count = first_window["airline_tier_distribution"]["mid_crew"]
        bag_count = first_window["airline_tier_distribution"]["batch_baggage"]
        mean_cpu = first_window["resource_metrics"]["mean_cpu_usage"]

        # Base scaling factors derived from Google cluster trace activity
        res_rate = round(max(10.0, 20.0 + (res_count * 4.5)), 2)
        crew_rate = round(max(5.0, 12.0 + (crew_count * 2.8)), 2)
        bag_rate = round(max(5.0, 15.0 + (bag_count * 3.2)), 2)
        gw_rate = round(res_rate + crew_rate + bag_rate, 2)

        # Baseline latency modulated by cluster CPU pressure
        res_lat = round(15.0 + (mean_cpu * 35.0), 2)
        crew_lat = round(18.0 + (mean_cpu * 40.0), 2)
        bag_lat = round(12.0 + (mean_cpu * 25.0), 2)
        gw_lat = round(35.0 + (mean_cpu * 60.0), 2)

        # Error rates triggered if evictions or failures occurred in trace
        fail_evict = first_window["task_event_counts"]["fail"] + first_window["task_event_counts"]["evict"]
        err_rate = round(min(0.20, fail_evict * 0.02), 4)

        return {
            "reservations": {
                "request_rate": res_rate,
                "latency_ms": res_lat,
                "error_rate": 0.0 if res_count > 0 else err_rate,
                "cpu_load": round(min(1.0, 0.2 + mean_cpu * 0.8), 3),
            },
            "crew": {
                "request_rate": crew_rate,
                "latency_ms": crew_lat,
                "error_rate": err_rate * 0.5,
                "cpu_load": round(min(1.0, 0.15 + mean_cpu * 0.7), 3),
            },
            "baggage": {
                "request_rate": bag_rate,
                "latency_ms": bag_lat,
                "error_rate": err_rate,
                "cpu_load": round(min(1.0, 0.1 + mean_cpu * 0.6), 3),
            },
            "boundary-gateway": {
                "request_rate": gw_rate,
                "latency_ms": gw_lat,
                "error_rate": err_rate * 0.75,
                "cpu_load": round(min(1.0, 0.3 + mean_cpu * 1.2), 3),
            },
            "legacy-core": {
                "request_rate": gw_rate,
                "latency_ms": round(gw_lat + 25.0, 2),
                "error_rate": 0.0,
                "cpu_load": round(min(1.0, 0.35 + mean_cpu * 1.1), 3),
            },
        }
