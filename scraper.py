import arxiv
import tarfile
import os
import requests
import json
import time
from tqdm import tqdm
from stats import StatsCollector, file_size, dir_size
import random
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import argparse
import multiprocessing

_LAST_S2_TIME = 0
S2_HEADERS = {"User-Agent": "23120318-scraper"}

# Timing configuration (can be changed via CLI in `main()`)
# Defaults tuned to be polite to services; reduce if you want faster local measurements.
ARXIV_WAIT = 3.1        # wait between arXiv metadata/source calls
S2_MIN_WAIT = 1.1       # minimum spacing between Semantic Scholar requests
S2_POST_SLEEP = 0.6     # short polite sleep between S2 POST chunks
WINDOW_SLEEP = 1.0      # sleep between windows of processing

def s2_wait_before_request():
    """Simple spacing between S2 requests to avoid bursts (1.1s).
    Kept minimal to keep code easy to understand.
    """
    global _LAST_S2_TIME
    now = time.monotonic()
    elapsed = now - _LAST_S2_TIME
    if elapsed < S2_MIN_WAIT:
        time.sleep(S2_MIN_WAIT - elapsed)
    _LAST_S2_TIME = time.monotonic()


def arxiv_fetch_with_retry(client, search_obj, max_attempts=5, base_wait=2, max_wait=60):
    """Fetch the first result from an arxiv.Search with retries on transient errors.

    This wraps next(client.results(search_obj)) and will retry on transient
    network errors or when the underlying library surfaces rate-limit
    errors (common symptom: exception message mentioning 429 or "rate").
    """
    attempts = 0
    last_exc = None
    while attempts < max_attempts:
        try:
            return next(client.results(search_obj))
        except StopIteration:
            # no result for this id — caller handles it
            raise
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # consider 429/rate errors or network timeouts transient
            is_rate = "429" in msg or "rate" in msg or "too many" in msg
            is_net = isinstance(e, requests.RequestException) or "timeout" in msg or "timed out" in msg
            if not (is_rate or is_net):
                # non-transient
                raise

            # exponential backoff with jitter
            wait = min(max_wait, base_wait * (2 ** attempts))
            jitter = 0.8 + random.random() * 0.4
            wait = max(1, int(wait * jitter))
            tqdm.write(f"    WARNING: arXiv fetch transient error (attempt {attempts+1}/{max_attempts}), backing off {wait}s: {e}")
            time.sleep(wait)
            attempts += 1

    # exhausted attempts
    raise Exception(f"arXiv fetch failed after {max_attempts} attempts: last error: {last_exc}")

# removed complex window counting for clarity

def decompress_and_filter(tar_path, extract_path):
    """Decompress either a tar.gz archive or a single-file .gz containing
    a TeX/BibTeX source. Returns True if .tex/.bib files were extracted.

    The function is defensive: it will silently return False for blobs
    that are not tar/gzip or that do not contain recognizable source.
    """

    # Ensure target dir exists
    os.makedirs(extract_path, exist_ok=True)

    # 1) Try to open as a tar.gz archive
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
    except Exception:
        # 2) Not a tar.gz (or extraction failed). Try single-file gzip.
        try:
            with gzip.open(tar_path, 'rb') as gz:
                sample = gz.read(8192)
        except Exception:
            # invalid gzip header or unreadable -> treat as not extractable
            return False

        # quick heuristics to detect TeX/BibTeX or PDF
        is_tex = b"\\begin{document}" in sample or b"\\documentclass" in sample
        is_bib = b"@article{" in sample or b"@misc{" in sample or b"@book{" in sample or b"@inproceedings{" in sample or b"\\bibitem" in sample
        is_pdf = sample.startswith(b"%PDF")

        if is_pdf:
            # it's a PDF, not source
            return False

        if not (is_tex or is_bib):
            # not recognizable as source
            return False

        # It's likely a single-source gzip; try to preserve the original
        # filename stored in the gzip header (FNAME) if present. Fall back
        # to the downloaded base name (without .gz) when the header lacks
        # a name. Do not force a .tex/.bib extension.
        out_name = None
        try:
            with open(tar_path, "rb") as fh:
                hdr = fh.read(10)
                if len(hdr) >= 10 and hdr[0:2] == b"\x1f\x8b":
                    flg = hdr[3]
                    pos = 10
                    # if FEXTRA set, skip XLEN and extra field
                    if flg & 0x04:
                        fh.seek(pos)
                        xlen_bytes = fh.read(2)
                        if len(xlen_bytes) == 2:
                            xlen = int.from_bytes(xlen_bytes, "little")
                            pos += 2 + xlen
                    fh.seek(pos)
                    if flg & 0x08:
                        # read original filename (null-terminated)
                        name_bytes = bytearray()
                        while True:
                            b = fh.read(1)
                            if not b or b == b"\x00":
                                break
                            name_bytes.extend(b)
                        try:
                            out_name = name_bytes.decode("utf-8", errors="replace")
                        except Exception:
                            out_name = name_bytes.decode("latin-1", errors="replace")
        except Exception:
            out_name = None

        if not out_name:
            base_name = os.path.basename(tar_path)
            if base_name.endswith(".tar.gz"):
                out_name = base_name[:-7]
            elif base_name.endswith(".gz"):
                out_name = base_name[:-3]
            else:
                out_name = base_name

        out_path = os.path.join(extract_path, out_name)

        try:
            # reopen and stream decompressed contents to disk
            with gzip.open(tar_path, "rb") as gz, open(out_path, "wb") as out_f:
                shutil.copyfileobj(gz, out_f)
        except Exception:
            return False

    # After extraction, remove non-.tex/.bib files and empty dirs
    allowed_extensions = {".tex", ".bib"}
    found_allowed = False
    for root, dirs, files in os.walk(extract_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if not any(name.endswith(ext) for ext in allowed_extensions):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            else:
                found_allowed = True

        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except OSError:
                pass

    return found_allowed

def get_versions(arxiv_id, tex_dir, stats=None, download_workers=1, decompress_workers=1):
    client = arxiv.Client()

    search_latest = arxiv.Search(id_list=[arxiv_id])
    
    try:
        latest_paper = arxiv_fetch_with_retry(client, search_latest)
    except StopIteration:
        tqdm.write(f"    ERROR: Paper ID {arxiv_id} not found.")
        raise
    except Exception as e:
        tqdm.write(f"    ERROR: Failed to fetch paper. Reason: {e}")
        raise
    time.sleep(ARXIV_WAIT)

    # Parse latest version number from entry_id (e.g., 'http://arxiv.org/abs/2409.05017v2')
    try:
        latest_version_num = int(latest_paper.entry_id.split("v")[-1]) 
    except (ValueError, IndexError):
        tqdm.write(f"    ERROR: Could not parse version number from entry_id: {latest_paper.entry_id}")
        raise

    metadata = {
        "paper_title": latest_paper.title,
        "authors": [str(author) for author in latest_paper.authors],
        "submission_date": None, # Will get this from v1
        "revised_dates": [],
        "publication_venue": latest_paper.journal_ref,
        "arxiv_id": arxiv_id,
        "total_versions": latest_version_num
    }
    
    # Build list of versions to fetch
    versions = list(range(1, latest_version_num + 1))

    # --- Parallel path: download multiple version blobs concurrently (I/O bound)
    version_infos = []
    for v in versions:
        # query id uses dot (for arXiv), filesystem tag uses hyphen
        query_version_id = f"{arxiv_id}v{v}"
        fs_version_tag = f"{arxiv_id.replace(".", "-")}v{v}"
        tar_filename = f"{query_version_id}.tar.gz"
        tar_path = os.path.join(tex_dir, tar_filename)
        extract_path = os.path.join(tex_dir, fs_version_tag)
        os.makedirs(extract_path, exist_ok=True)
        # store v, query id, fs tag, tar_path, extract_path
        version_infos.append((v, query_version_id, fs_version_tag, tar_path, extract_path))

    headers = dict(S2_HEADERS)

    def _download_version(version_id_str, tar_path):
        source_url = f"https://arxiv.org/e-print/{version_id_str}"
        try:
            with requests.get(source_url, stream=True, allow_redirects=True, timeout=30, headers=headers) as resp:
                if resp.status_code != 200:
                    return (version_id_str, tar_path, False, resp.status_code)
                try:
                    with open(tar_path, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1024*64):
                            if chunk:
                                fh.write(chunk)
                except Exception:
                    if os.path.exists(tar_path):
                        try:
                            os.remove(tar_path)
                        except Exception:
                            pass
                    return (version_id_str, tar_path, False, "write_error")
        except requests.RequestException:
            return (version_id_str, tar_path, False, "request_error")

        # small polite jitter to avoid tight bursts when many threads finish
        try:
            time.sleep(0.15 + random.random() * 0.1)
        except Exception:
            pass

        return (version_id_str, tar_path, True, 200)

    downloaded = []
    with ThreadPoolExecutor(max_workers=max(1, download_workers)) as tpool:
        futures = {tpool.submit(_download_version, vi[1], vi[3]): vi for vi in version_infos}
        for fut in as_completed(futures):
            vi = futures[fut]
            v, query_version_id, fs_version_tag, tar_path, extract_path = vi
            try:
                ver_id, path, ok, info = fut.result()
            except Exception as e:
                tqdm.write(f"    WARNING: Download task for {query_version_id} raised: {e}")
                ok = False
                path = tar_path
            if ok and os.path.exists(path):
                # record pre-extract size
                try:
                    if stats is not None:
                        stats.add_size_before(file_size(path))
                except OSError:
                    pass
                downloaded.append((query_version_id, path, extract_path))

    # Decompress in parallel if requested
    decompressed_ok = {}
    if decompress_workers and decompress_workers > 1:
        with ProcessPoolExecutor(max_workers=max(1, decompress_workers)) as ppool:
            futures = {ppool.submit(decompress_and_filter, path, extract_path): (query_version_id, path, extract_path) for (query_version_id, path, extract_path) in downloaded}
            for fut in as_completed(futures):
                query_version_id, path, extract_path = futures[fut]
                try:
                    ok = fut.result()
                except Exception as e:
                    tqdm.write(f"    WARNING: Decompress task for {query_version_id} failed: {e}")
                    ok = False
                decompressed_ok[query_version_id] = (ok, path, extract_path)
    else:
        # sequential decompression
        for query_version_id, path, extract_path in downloaded:
            try:
                ok = decompress_and_filter(path, extract_path)
            except Exception as e:
                tqdm.write(f"    WARNING: Decompress for {query_version_id} failed: {e}")
                ok = False
            decompressed_ok[query_version_id] = (ok, path, extract_path)

    # Update metadata, stats and cleanup
    for v, query_version_id, fs_version_tag, tar_path, extract_path in version_infos:
        # fetch metadata per-version (still sequential to keep original sleeps and rate-limits)
        search_version = arxiv.Search(id_list=[query_version_id])
        try:
            paper_version = arxiv_fetch_with_retry(client, search_version)
        except Exception:
            # skip metadata only; but we keep whatever extraction happened
            paper_version = None
        time.sleep(ARXIV_WAIT)

        if v == 1 and paper_version:
            metadata["submission_date"] = paper_version.published.strftime("%Y-%m-%d")
            metadata["revised_dates"].append(paper_version.published.strftime("%Y-%m-%d"))
        elif paper_version:
            metadata["revised_dates"].append(paper_version.updated.strftime("%Y-%m-%d"))

        info = decompressed_ok.get(query_version_id)
        if info and info[0]:
                try:
                    final_bytes = dir_size(extract_path)
                    if stats is not None:
                        stats.add_size_after(final_bytes)
                        stats.add_version(1)
                except OSError:
                    pass

        # remove tar file if present
        try:
            if os.path.exists(tar_path):
                os.remove(tar_path)
        except OSError as e:
            tqdm.write(f"    WARNING: Could not remove tarball: {tar_path}, {e}")

    return metadata

def get_and_process_references(arxiv_id, s2_cache=None, stats=None):
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    params = {
        "fields": "references.title,references.authors,references.externalIds,references.publicationDate,references.paperId"
    }

    # If we have a simple in-memory cache, return cached value immediately
    if s2_cache is not None and arxiv_id in s2_cache:
        return s2_cache.get(arxiv_id, {})

    # Simple single-request fallback (kept intentionally tiny and readable):
    try:
        s2_wait_before_request()
        response = requests.get(url, params=params, headers=S2_HEADERS, timeout=15)
        if response.status_code != 200:
            tqdm.write(f"    INFO: S2 returned HTTP {response.status_code} for {arxiv_id}. Treating as no references.")
            # record empty mapping to avoid retries
            if s2_cache is not None:
                try:
                    s2_cache[arxiv_id] = {}
                except Exception:
                    pass
            if stats is not None:
                try:
                    stats.add_ref_fetch_result(0, 0)
                except Exception:
                    pass
            return {}
        data = response.json()
        raw_count = len(data.get('references') or []) if isinstance(data, dict) else 0
        processed = process_s2_raw(arxiv_id, data, s2_cache=s2_cache)
        processed_count = len(processed) if processed else 0
        if stats is not None:
            try:
                stats.add_ref_fetch_result(raw_count, processed_count)
            except Exception:
                pass
        return processed
    except requests.RequestException as e:
        tqdm.write(f"    WARNING: S2 request error for {arxiv_id}: {e}. Skipping S2 for this paper.")
        # ensure we record an empty mapping to avoid repeated attempts
        if s2_cache is not None:
            try:
                s2_cache[arxiv_id] = {}
            except Exception:
                pass
        if stats is not None:
            try:
                stats.add_ref_fetch_result(0, 0)
            except Exception:
                pass
        return {}


def process_s2_raw(arxiv_id, data, s2_cache=None):
    """Normalize S2 paper JSON into {arXiv-id-formatted: metadata} mapping."""
    raw_references = data.get("references") if isinstance(data, dict) else None
    if not raw_references:
        return {}

    processed_references = {}
    for ref in raw_references:
        if not ref or not isinstance(ref, dict):
            continue
        external = ref.get("externalIds") or {}
        arxiv_raw = external.get("ArXiv")
        if not arxiv_raw:
            continue

        ref_arxiv_id_clean = str(arxiv_raw).split("v")[0]
        if "." not in ref_arxiv_id_clean:
            continue
        formatted_key = ref_arxiv_id_clean.replace(".", "-")

        authors = []
        for a in ref.get("authors", []) or []:
            if isinstance(a, dict) and a.get("name"):
                authors.append(a.get("name"))

        ref_metadata = {
            "paper_title": ref.get("title"),
            "authors": authors,
            "submission_date": ref.get("publicationDate"),
            "semantic_scholar_id": ref.get("paperId")
        }

        processed_references[formatted_key] = ref_metadata
    # store into cache if provided
    if s2_cache is not None:
        try:
            s2_cache[arxiv_id] = processed_references
        except Exception:
            pass

    return processed_references


def fetch_s2_batches(arxiv_ids, s2_cache, batch_size=100, progress=None, post_chunk_size=100, stats=None):
    """Fetch Semantic Scholar metadata for arXiv IDs using the batch API.
    Fill `s2_cache` with processed reference mappings for each id. Returns a
    short summary dict with counts for requested/success/empty/missing.
    """
    if not arxiv_ids:
        return

    # Only fetch IDs that are not already cached
    to_fetch = [aid for aid in arxiv_ids if aid not in s2_cache]
    if not to_fetch:
        return

    batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    fields = "references.title,references.authors,references.externalIds,references.publicationDate,references.paperId"
    params = {"fields": fields}
    headers = dict(S2_HEADERS)
    headers["Content-Type"] = "application/json"

    # Split into POST chunks to reduce per-request pressure.
    post_size = post_chunk_size or batch_size
    batches = [to_fetch[i:i + post_size] for i in range(0, len(to_fetch), post_size)]

    def fetch_batch(batch):
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            try:
                s2_wait_before_request()
                resp = requests.post(batch_url, json={"ids": [f"ARXIV:{b}" for b in batch]}, params=params, headers=headers, timeout=90)

                if resp.status_code == 429:
                    ra = resp.headers.get('Retry-After')
                    wait = None
                    if ra and ra.isdigit():
                        try:
                            wait = int(ra)
                        except Exception:
                            wait = None
                    if wait is None:
                        wait = min(60, 5 * (2 ** attempts))
                    else:
                        wait = min(wait, 60)
                    jitter = 0.8 + random.random() * 0.4
                    wait = max(1, int(wait * jitter))
                    tqdm.write(f"    WARNING: S2 batch 429. Backing off {int(wait)}s (attempt {attempts+1})")
                    time.sleep(wait)
                    attempts += 1
                    continue

                if resp.status_code in (503, 504):
                    wait = min(60, 5 * (2 ** attempts))
                    tqdm.write(f"    WARNING: S2 batch server error {resp.status_code}. Backing off {int(wait)}s")
                    time.sleep(wait)
                    attempts += 1
                    continue

                resp.raise_for_status()
                papers = resp.json()

                # papers is expected to be a list aligned with batch order
                for i, paper_dict in enumerate(papers):
                    aid = batch[i]
                    try:
                        raw_refs = 0
                        processed_count = 0
                        if paper_dict and isinstance(paper_dict, dict):
                            raw_refs = len(paper_dict.get("references") or [])
                        if paper_dict and paper_dict.get("paperId"):
                            processed = process_s2_raw(aid, paper_dict, s2_cache=s2_cache)
                            processed_count = len(processed) if processed else 0
                        else:
                            processed = {}
                            processed_count = 0

                        if s2_cache is not None:
                            try:
                                s2_cache[aid] = processed
                            except Exception:
                                pass

                        if stats is not None:
                            try:
                                stats.add_ref_fetch_result(raw_refs, processed_count)
                            except Exception:
                                pass
                    except Exception as e:
                        tqdm.write(f"    WARNING: Exception processing S2 response for {aid}: {e}")
                        if s2_cache is not None:
                            s2_cache[aid] = {}

                return True

            except requests.RequestException as e:
                wait = min(60, 5 * (2 ** attempts))
                jitter = 0.8 + random.random() * 0.4
                wait = max(1, int(wait * jitter))
                tqdm.write(f"    WARNING: Network error fetching S2 batch: {e}. Backing off {int(wait)}s")
                time.sleep(wait)
                attempts += 1
            except Exception as e:
                tqdm.write(f"    ERROR: Unexpected error in S2 batch fetch: {e}")
                # write empty entries for this batch
                if s2_cache is not None:
                    for b in batch:
                        try:
                            s2_cache[b] = {}
                        except Exception:
                            pass
                return False

        # exhausted attempts -> mark empties
        if s2_cache is not None:
            for b in batch:
                try:
                    s2_cache[b] = {}
                except Exception:
                    pass
        return False

    # Run POST chunks sequentially and collect quick stats
    success_count = 0
    empty_count = 0
    for batch in batches:
        try:
            fetch_batch(batch)
        except Exception as e:
            tqdm.write(f"    ERROR: S2 batch failed: {e}")

        # count results for this chunk
        for aid in batch:
            if aid in s2_cache:
                if s2_cache.get(aid):
                    success_count += 1
                else:
                    empty_count += 1

        # progress advance and a short polite sleep between POSTs
        if progress is not None:
            try:
                progress.update(len(batch))
            except Exception:
                pass
        try:
            time.sleep(S2_POST_SLEEP)
        except Exception:
            pass

    missing = len([aid for aid in to_fetch if aid not in s2_cache])
    return {"requested": len(to_fetch), "success": success_count, "empty": empty_count, "missing": missing}

def run_scraper(arxiv_ids, dir, prefetch_s2=True, s2_batch_size=100, download_workers=1, decompress_workers=1):
    print(f"Starting scraper. Output directory: {dir}")
    os.makedirs(dir, exist_ok=True)
    # initialize stats collector
    stats = StatsCollector()
    stats.start_total()
    # simple in-memory cache for Semantic Scholar results for this run
    s2_cache = {}


    # Process IDs in windows. For each window we fetch S2 metadata (batch) and
    # then immediately process those papers. This avoids prefetching the entire
    # list (which can overwhelm S2) while keeping batch benefits.
    if prefetch_s2:
        total_windows = (len(arxiv_ids) + s2_batch_size - 1) // s2_batch_size
        for widx in range(0, len(arxiv_ids), s2_batch_size):
            window = arxiv_ids[widx:widx + s2_batch_size]
            win_num = widx // s2_batch_size + 1

            tqdm.write(f"== Window {win_num}/{total_windows}: fetching S2 for {len(window)} IDs ==")
            # progress bar for the window's S2 fetch
            pbar_fetch = tqdm(total=len(window), desc=f"Fetching S2 {win_num}/{total_windows}", unit="paper")
            try:
                summary = fetch_s2_batches(window, s2_cache, batch_size=s2_batch_size, progress=pbar_fetch, post_chunk_size=100, stats=stats)
            except Exception as e:
                tqdm.write(f"WARNING: S2 batch prefetch for window {win_num} failed: {e}")
                summary = None
            finally:
                pbar_fetch.close()

            # after fetch, report summary (use returned summary if available)
            if summary:
                tqdm.write(f"== Window {win_num}/{total_windows}: fetched {summary['success']}/{summary['requested']} papers from S2 ({summary['empty']} empty, {summary['missing']} missing) ==")
            else:
                success_count = sum(1 for aid in window if s2_cache.get(aid))
                tqdm.write(f"== Window {win_num}/{total_windows}: fetched {success_count}/{len(window)} papers from S2 ==")

            pbar = tqdm(window, desc=f"Window {win_num}/{total_windows}", unit="paper")
            for arxiv_id in pbar:
                pbar.set_description(f"Processing {arxiv_id}")
                try:
                    stats.start_paper()
                    paper_dir_name = arxiv_id.replace(".", "-") 
                    paper_path = os.path.join(dir, paper_dir_name)
                    tex_path = os.path.join(paper_path, "tex")
                    os.makedirs(tex_path, exist_ok=True)

                    # Download all versions (optionally in parallel), filter, and get metadata
                    metadata = get_versions(arxiv_id, tex_path, stats=stats, download_workers=download_workers, decompress_workers=decompress_workers)
                    metadata_path = os.path.join(paper_path, "metadata.json")
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=4)

                    # Get and write references from cache (batch prefetch filled s2_cache)
                    try:
                        references = get_and_process_references(arxiv_id, s2_cache=s2_cache, stats=stats) or {}
                        # update stats; ignore minor filesystem errors only
                        try:
                            stats.add_refs(len(references))
                        except OSError:
                            pass
                        references_path = os.path.join(paper_path, "references.json")
                        with open(references_path, "w", encoding="utf-8") as f:
                            json.dump(references, f, ensure_ascii=False, indent=4)
                    except Exception as e:
                        tqdm.write(f"    WARNING: Failed to write references for {arxiv_id}: {e}")

                    stats.end_paper(success=True)

                except Exception as e:
                    tqdm.write(f"!!! FATAL ERROR processing {arxiv_id}: {e}")
                    tqdm.write("!!! Skipping this ID and moving to the next one.")
                    stats.end_paper(success=False)
                    continue

            # small sleep between windows to be polite to S2/arXiv
            time.sleep(WINDOW_SLEEP)
    else:
        # No prefetch: process sequentially and fetch S2 per-paper when needed
        pbar = tqdm(arxiv_ids, desc="Scraping papers...", unit="paper")
        for arxiv_id in pbar:
            pbar.set_description(f"Processing {arxiv_id}")
            try:
                stats.start_paper()
                paper_dir_name = arxiv_id.replace(".", "-") 
                paper_path = os.path.join(dir, paper_dir_name)
                tex_path = os.path.join(paper_path, "tex")
                os.makedirs(tex_path, exist_ok=True)

                metadata = get_versions(arxiv_id, tex_path, stats=stats, download_workers=download_workers, decompress_workers=decompress_workers)
                metadata_path = os.path.join(paper_path, "metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)

                try:
                    references = get_and_process_references(arxiv_id, s2_cache=s2_cache, stats=stats) or {}
                    try:
                        stats.add_refs(len(references))
                    except OSError:
                        pass
                    references_path = os.path.join(paper_path, "references.json")
                    with open(references_path, "w", encoding="utf-8") as f:
                        json.dump(references, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    tqdm.write(f"    WARNING: Failed to write references for {arxiv_id}: {e}")

                stats.end_paper(success=True)

            except Exception as e:
                tqdm.write(f"!!! FATAL ERROR processing {arxiv_id}: {e}")
                tqdm.write("!!! Skipping this ID and moving to the next one.")
                stats.end_paper(success=False)
                continue
    # No deferred-pass: batch prefetch should have filled s2_cache for missing IDs

    # finalize stats and save report
    stats.end_total()
    # Save report_stats.json outside the per-run output folder (one level up)
    out_abs = os.path.abspath(dir)
    parent_dir = os.path.dirname(out_abs) or out_abs
    stats_path = os.path.join(parent_dir, "report_stats.json")
    try:
        stats.save(stats_path)
        tqdm.write(f"Saved stats to {stats_path}")
    except Exception as e:
        tqdm.write(f"    WARNING: Failed to save stats to {stats_path}: {e}")
    # no persistent S2 cache file: we keep an in-memory cache for this run

def main():
    parser = argparse.ArgumentParser(description="Scrape arXiv sources and references (parallel-ready).")
    parser.add_argument("--outdir", "-o", default="23120318", help="Output directory")
    parser.add_argument("--year-month", nargs="?", default="2409", help="Year and month code for arXiv IDs (e.g., '2409')")
    parser.add_argument("--start", type=int, default=5017, help="Start index for generated arXiv IDs (inclusive)")
    parser.add_argument("--end", type=int, default=10016, help="End index for generated arXiv IDs (inclusive)")
    parser.add_argument("--download-workers", type=int, default=None, help="Number of parallel download worker threads (I/O)")
    parser.add_argument("--decompress-workers", type=int, default=None, help="Number of parallel decompress worker processes (CPU)")
    parser.add_argument("--no-prefetch-s2", dest="prefetch_s2", action="store_false", help="Disable S2 prefetching per-window")
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false", help="Disable fast mode (fast defaults enabled by default)")
    # fast mode enabled by default; provide --no-fast-mode to turn it off
    parser.set_defaults(fast_mode=True)
    parser.add_argument("--arxiv-wait", type=float, default=None, help="Seconds to wait between arXiv metadata/source calls (overrides default)")
    parser.add_argument("--s2-wait", type=float, default=None, help="Minimum seconds between S2 requests (overrides default)")
    parser.add_argument("--s2-post-sleep", type=float, default=None, help="Seconds to sleep between S2 POST chunks (overrides default)")
    parser.add_argument("--window-sleep", type=float, default=None, help="Seconds to sleep between windows of processing (overrides default)")
    args = parser.parse_args()

    DIR = args.outdir
    YEAR_MONTH = args.year_month
    START_ID = args.start
    END_ID = args.end

    cpu_count = multiprocessing.cpu_count() or 2
    # sensible defaults: a few download threads and cpu-bound decompress processes
    download_workers = args.download_workers if args.download_workers is not None else min(8, max(2, cpu_count * 2))
    decompress_workers = args.decompress_workers if args.decompress_workers is not None else max(1, min(cpu_count, cpu_count - 1 if cpu_count > 1 else 1))

    # Fast-mode tuning: increase workers and reduce waits when requested
    fast_mode = bool(args.fast_mode)
    if fast_mode:
        if args.download_workers is None:
            download_workers = min(16, max(4, cpu_count * 2))
        if args.decompress_workers is None:
            decompress_workers = min(8, max(1, cpu_count))


    ids = range(START_ID, END_ID + 1)
    arxiv_ids_to_scrape = [f"{YEAR_MONTH}.{str(i).zfill(5)}" for i in ids]

    # Apply optional CLI overrides to timing globals, respecting fast-mode defaults
    global ARXIV_WAIT, S2_MIN_WAIT, S2_POST_SLEEP, WINDOW_SLEEP
    # If fast-mode is enabled and user didn't explicitly pass wait overrides,
    # apply faster defaults first, then let explicit args override them.
    if fast_mode:
        if args.arxiv_wait is None:
            ARXIV_WAIT = 0.25
        if args.s2_wait is None:
            S2_MIN_WAIT = 0.18
        if args.s2_post_sleep is None:
            S2_POST_SLEEP = 0.05
        if args.window_sleep is None:
            WINDOW_SLEEP = 0.1

    if args.arxiv_wait is not None:
        ARXIV_WAIT = float(args.arxiv_wait)
    if args.s2_wait is not None:
        S2_MIN_WAIT = float(args.s2_wait)
    if args.s2_post_sleep is not None:
        S2_POST_SLEEP = float(args.s2_post_sleep)
    if args.window_sleep is not None:
        WINDOW_SLEEP = float(args.window_sleep)

    print(f"Generated {len(arxiv_ids_to_scrape)} IDs to scrape.")
    print(f"From {arxiv_ids_to_scrape[0]} to {arxiv_ids_to_scrape[-1]}")
    print(f"Workers: download={download_workers}, decompress={decompress_workers}")

    # Decide S2 batch size: larger batches amortize requests (good for fast-mode)
    s2_batch_size = 200 if fast_mode else 100
    try:
        run_scraper(arxiv_ids_to_scrape, DIR, prefetch_s2=args.prefetch_s2, s2_batch_size=s2_batch_size, download_workers=download_workers, decompress_workers=decompress_workers)
    except KeyboardInterrupt:
        tqdm.write("\nInterrupted by user. Exiting.")
    except Exception as e:
        tqdm.write(f"FATAL: Scraper failed: {e}")
    finally:
        print("\nScraping process finished.")

if __name__ == "__main__":
    main()