import time
import threading
import json
import os
try:
    import psutil
except Exception:
    psutil = None


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def dir_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class StatsCollector:
    """Collect runtime and resource usage statistics.

    Usage:
      stats = StatsCollector()
      stats.start_total()
      stats.start_paper()
      ... per-paper events ...
      stats.end_paper(success=True)
      stats.end_total()
      stats.save(path)
    """

    def __init__(self, poll_interval=0.5):
        self.start_time = None
        self.end_time = None
        self.total_papers = 0
        self.success_papers = 0
        self.total_versions = 0
        self.total_refs = 0
        self.total_refs_raw = 0
        self.total_refs_with_meta = 0
        self.ref_fetch_requests = 0
        self.sizes_before = []
        self.sizes_after = []
        self.per_paper_times = []
        self._poll_interval = poll_interval
        self.peak_rss = 0
        self._proc = psutil.Process() if psutil is not None else None
        self._stop_event = threading.Event()
        self._poll_thread = None

    def start_total(self):
        self.start_time = time.perf_counter()
        if self._proc is not None:
            self._stop_event.clear()
            self._poll_thread = threading.Thread(target=self._poll_memory, daemon=True)
            self._poll_thread.start()

    def end_total(self):
        self.end_time = time.perf_counter()
        if self._proc is not None:
            self._stop_event.set()
            if self._poll_thread:
                self._poll_thread.join(timeout=2)

    def _poll_memory(self):
        while not self._stop_event.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self._poll_interval)

    def start_paper(self):
        self._paper_start = time.perf_counter()

    def end_paper(self, success=True):
        elapsed = time.perf_counter() - getattr(self, '_paper_start', time.perf_counter())
        self.per_paper_times.append(elapsed)
        self.total_papers += 1
        if success:
            self.success_papers += 1

    def add_version(self, count=1):
        self.total_versions += count

    def add_refs(self, n):
        try:
            self.total_refs += int(n)
        except Exception:
            pass

    def add_ref_fetch_result(self, raw_count, processed_count):
        """Record results from a single S2 fetch for one paper.

        raw_count: number of references returned in the raw S2 response
        processed_count: number of references for which we kept metadata
        """
        try:
            self.total_refs_raw += int(raw_count or 0)
        except Exception:
            pass
        try:
            self.total_refs_with_meta += int(processed_count or 0)
        except Exception:
            pass
        try:
            self.ref_fetch_requests += 1
        except Exception:
            pass

    def add_size_before(self, bytes_size):
        try:
            self.sizes_before.append(int(bytes_size))
        except Exception:
            pass

    def add_size_after(self, bytes_size):
        try:
            self.sizes_after.append(int(bytes_size))
        except Exception:
            pass

    def summary(self):
        total_time = (self.end_time - self.start_time) if (self.start_time and self.end_time) else None
        avg_time = (sum(self.per_paper_times) / len(self.per_paper_times)) if self.per_paper_times else None
        avg_before = (sum(self.sizes_before) / len(self.sizes_before)) if self.sizes_before else 0
        avg_after = (sum(self.sizes_after) / len(self.sizes_after)) if self.sizes_after else 0
        avg_refs_per_paper = (self.total_refs / self.total_papers) if self.total_papers else 0
        ref_meta_success_rate = None
        if self.total_refs_raw:
            ref_meta_success_rate = float(self.total_refs_with_meta) / float(self.total_refs_raw)
        elif self.total_refs_with_meta:
            ref_meta_success_rate = 1.0
        return {
            "total_time_s": total_time,
            "total_papers": self.total_papers,
            "success_papers": self.success_papers,
            "success_rate": (float(self.success_papers) / float(self.total_papers)) if self.total_papers else None,
            "total_versions": self.total_versions,
            "total_references": self.total_refs,
            "total_references_raw": self.total_refs_raw,
            "total_references_with_metadata": self.total_refs_with_meta,
            "avg_references_per_paper": avg_refs_per_paper,
            "ref_metadata_success_rate": ref_meta_success_rate,
            "avg_time_per_paper_s": avg_time,
            "peak_rss_bytes": self.peak_rss,
            "avg_tar_size_bytes": avg_before,
            "avg_final_tex_size_bytes": avg_after
        }

    def save(self, path="report_demo.json"):
        data = self.summary()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
