import argparse
import datetime as dt
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from venue_pattern import VENUE_PATTERNS

import openreview
from tqdm import tqdm

API_BASEURL = "https://api2.openreview.net"

# --------------------- Rate limiting --------------------

class RequestBudget:
    """
    A simple call counter so we can sleep every `sleep_every` calls.
    This counts *our* client method invocations (paged), which is what we control.
    """
    def __init__(self, sleep_every: int = 200, sleep_seconds: int = 15):
        self.sleep_every = max(1, sleep_every)
        self.sleep_seconds = max(1, sleep_seconds)
        self.calls = 0

    def tick(self, label: str = ""):
        self.calls += 1
        if self.calls % self.sleep_every == 0:
            # periodic nap to avoid hitting the server-side sliding window
            time.sleep(self.sleep_seconds)

def sleep_backoff(attempt: int, base: float = 3.0, cap: float = 30.0):
    """Exponential-ish backoff with a soft cap."""
    delay = min(cap, base * (1.7 ** (attempt - 1)))
    time.sleep(delay)

def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return "Too many requests" in msg or "RateLimitError" in msg or "status': 429" in msg or " 429 " in msg

def safe_call(budget: RequestBudget, fn, *args, retries: int = 6, **kwargs):
    """
    Wrap any OpenReview call, count it toward our budget, and retry on failure.
    - Sleeps on *any* exception (rate limit or otherwise).
    - Uses exponential backoff across attempts.
    """
    attempt = 1
    while True:
        try:
            budget.tick(fn.__name__)
            return fn(*args, **kwargs)
        except Exception as e:
            # Sleep a bit extra if it's the explicit rate-limit case
            if is_rate_limit_error(e):
                # small bonus nap; the server hints a reset time, but we keep it simple and robust
                time.sleep(15)
            if attempt >= retries:
                raise
            sleep_backoff(attempt)
            attempt += 1

# ------------------------ Helpers -----------------------

def years_list(n_back: int, end_year: Optional[int] = None) -> List[int]:
    y = end_year or dt.datetime.utcnow().year
    return list(range(y, y - n_back, -1))

def normalize_keywords(kw: Any) -> List[str]:
    if kw is None:
        return []
    if isinstance(kw, list):
        return [str(k).strip() for k in kw if k and str(k).strip()]
    if isinstance(kw, str):
        return [k.strip() for k in kw.split(",") if k.strip()]
    return []

def pick_preferred_name(profile: "openreview.Profile") -> Optional[str]:
    try:
        return profile.get_preferred_name(pretty=True)
    except Exception:
        # Fallbacks via the raw content names
        names = (profile.content or {}).get("names") or []
        for n in names:
            if n.get("preferred", False):
                return n.get("first", "") + " " + n.get("last", "")
        if names:
            n0 = names[0]
            return (n0.get("first", "") + " " + n0.get("last", "")).strip() or None
        return None

def extract_history(profile: "openreview.Profile") -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (history_list, current_affiliation) extracted from profile.content['history'].

    Each history entry contains:
      - start, end (if available)
      - position/title (position preferred; fallback to title)
      - department (if any)
      - institution: name, domain, country (if available)
      - location (if present)

    current_affiliation = first item with no 'end'; if none, None.
    """
    content = profile.content or {}
    raw_hist = content.get("history") or []
    hist: List[Dict[str, Any]] = []
    for h in raw_hist:
        inst = (h.get("institution") or {})
        hist.append({
            "start": h.get("start"),
            "end": h.get("end"),
            "position": h.get("position") or h.get("title"),
            "department": h.get("department"),
            "location": h.get("location"),
            "institution": {
                "name": inst.get("name"),
                "domain": inst.get("domain"),
                "country": inst.get("country"),
            }
        })
    return hist

def note_to_paper_dict(note: "openreview.api.Note") -> Dict[str, Any]:
    c = note.content or {}
    getv = lambda key, default="": (c.get(key, {}) or {}).get("value", default)

    return {
        "paper_id": note.id,
        "forum": note.forum,
        "venueid": getv("venueid", ""),
        "venue": getv("venue", ""),
        "title": getv("title", ""),
        "abstract": getv("abstract", ""),
        "keywords": normalize_keywords(getv("keywords", [])),
        "authors": getv("authors", []),        # display names (list)
        "authorids": getv("authorids", []),    # tilde ids / emails (list)
    }

# -------------------- Core fetchers ---------------------

def fetch_notes_paginated(client: openreview.api.OpenReviewClient,
                          budget: RequestBudget,
                          venue_id: str,
                          page_size: int = 100) -> List["openreview.api.Note"]:
    """
    Explicitly paginates get_notes so we can count/sleep every page.
    """
    out: List["openreview.api.Note"] = []
    offset = 0
    while True:
        batch = safe_call(
            budget,
            client.get_notes,
            content={"venueid": venue_id},
            limit=page_size,
            offset=offset,
        )
        if not batch:
            break
        out.extend(batch)
        offset += len(batch)
        # light pacing between pages
        time.sleep(0.05)
    return out

def chunked(iterable: Iterable[Any], n: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for x in iterable:
        chunk.append(x)
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def fetch_profiles_dict(client: openreview.api.OpenReviewClient,
                        budget: RequestBudget,
                        author_ids: List[str],
                        batch_size: int = 100) -> Dict[str, Optional["openreview.Profile"]]:
    """
    Resolves author ids/emails to Profile objects. Returns a dict keyed by the *original* id/email.
    We chunk requests to keep call counts low and tolerate rate limits.
    """
    from openreview import tools as or_tools

    results: Dict[str, Optional["openreview.Profile"]] = {}
    # OpenReview sometimes has duplicates or blank ids; clean first
    cleaned = [a for a in dict.fromkeys(author_ids) if a]

    for block in tqdm(list(chunked(cleaned, batch_size)), desc="Resolving profiles", unit="block"):
        # The helper may issue several API calls internally; we wrap & backoff.
        profiles = safe_call(
            budget,
            or_tools.get_profiles,
            client,
            block,
            as_dict=True   # -> maps key to Profile or None
        )
        # Merge
        for k in block:
            results[k] = profiles.get(k)
        # Gentle pacing between blocks
        time.sleep(0.1)

    return results

# -------------------- Orchestration --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3, help="Number of recent years to fetch (including current year).")
    parser.add_argument("--venues", nargs="+", default=["ICLR", "ICML", "NeurIPS", "COLM", "TMLR"],
                        help="Subset of venues to include. Supported venues are listed in venue_pattern.py")
    parser.add_argument("--page-size", type=int, default=100, help="API page size for notes pagination.")
    parser.add_argument("--sleep-every", type=int, default=200, help="Sleep every N client calls (to avoid 429).")
    parser.add_argument("--sleep-seconds", type=int, default=15, help="Seconds to sleep on periodic naps & 429s.")
    parser.add_argument("--out", type=str, default="openreview_output", help="Path to output JSON.")
    args = parser.parse_args()

    # Instantiate client
    client = openreview.api.OpenReviewClient(baseurl=API_BASEURL,
                                             token=os.environ.get("OPENREVIEW_TOKEN"),)

    # Request-budget tracker
    budget = RequestBudget(sleep_every=args.sleep_every, sleep_seconds=args.sleep_seconds)

    # Build target venueids
    yrs = years_list(args.years)
    targets: List[str] = []
    for v in args.venues:
        if v not in VENUE_PATTERNS:
            raise ValueError(f"Unknown venue key: {v}. Allowed: {list(VENUE_PATTERNS.keys())}")
        for y in yrs:
            targets.append(VENUE_PATTERNS[v].format(year=y))
    targets = set(targets)

    # 1) Pull papers (Notes) per venue
    papers: List[Dict[str, Any]] = []
    print(f"Fetching accepted/published submissions for {len(targets)} venue-years...")
    for vid in tqdm(targets, desc="Venues", unit="venue"):
        notes = fetch_notes_paginated(client, budget, vid, page_size=args.page_size)
        for n in notes:
            papers.append(note_to_paper_dict(n))

    # 2) Collect all authorids (tilde ids or emails)
    all_author_ids: List[str] = []
    for p in papers:
        all_author_ids.extend(p.get("authorids", []) or [])
    # 3) Resolve profiles
    profiles_by_key = fetch_profiles_dict(client, budget, all_author_ids, batch_size=100)

    # 4) Build authors list (unique), each with history/current affiliation and list of their papers
    authors_map: Dict[str, Dict[str, Any]] = {}  # key: canonical id (profile.id or fallback raw id)

    def ensure_author_entry(raw_key: str, display_name: Optional[str]) -> str:
        prof = profiles_by_key.get(raw_key)
        if prof:
            canonical_key = prof.id
            if canonical_key not in authors_map:
                hist = extract_history(prof)
                authors_map[canonical_key] = {
                    "profile_id": prof.id,
                    "profile_url": f"https://openreview.net/profile?id={prof.id}",
                    "name": pick_preferred_name(prof) or display_name,
                    "orcid": (prof.content or {}).get("orcid"),
                    "history": hist,
                    "papers": [],  # will append
                }
            return canonical_key
        else:
            # No profile claimed; keep what we can
            canonical_key = raw_key
            if canonical_key not in authors_map:
                authors_map[canonical_key] = {
                    "profile_id": None,
                    "unclaimed_id": raw_key,  # typically an email or string
                    "profile_url": None,
                    "name": display_name,
                    "orcid": None,
                    "history": [],
                    "current_affiliation": None,
                    "papers": [],
                }
            return canonical_key

    # 5) Link papers to authors
    for p in papers:
        names = p.get("authors", []) or []
        ids   = p.get("authorids", []) or []
        # pair up defensively
        max_len = max(len(names), len(ids))
        for i in range(max_len):
            name_i = names[i] if i < len(names) else None
            id_i   = ids[i] if i < len(ids) else (names[i] if i < len(names) else None)
            if not id_i and not name_i:
                continue
            key = ensure_author_entry(id_i or name_i, name_i)
            authors_map[key]["papers"].append({
                "paper_id": p["paper_id"],
                "title": p["title"],
                "venueid": p["venueid"],
                "venue": p["venue"],
            })

    # 6) Write output
    authors = list(authors_map.values())
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'papers.json'), "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, 'authors.json'), "w", encoding="utf-8") as f:
        json.dump(authors, f, ensure_ascii=False, indent=1)

    print(f"Done. Wrote {len(papers)} papers and {len(authors)} authors to: {args.out}")

if __name__ == "__main__":
    main()
