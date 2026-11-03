"""Process resource fields shared by workflow tasks and private benchmark trials."""
import resource
import time


def process_resources(started):
    """Elapsed wall time, cumulative CPU including children, and peak RSS in KiB (Linux)."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"wall_s": time.perf_counter() - started,
            "cpu_s": own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime,
            "peak_rss_kib": max(own.ru_maxrss, children.ru_maxrss)}
