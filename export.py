import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from jinja2 import Environment, FileSystemLoader, select_autoescape


def load_payload(path: str) -> Dict[str, Any]:
    return {'papers': json.load(open(os.path.join(path, 'papers.json'), encoding='utf-8')),
            'authors': json.load(open(os.path.join(path, 'authors.json'), encoding='utf-8'))}


def norm_text(x: Optional[str]) -> str:
    x = (x or "").strip()
    x = re.sub(r"-", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x


def paper_matches(p: Dict[str, Any], terms: List[str]) -> bool:
    """Match if ANY search term is in title/abstract/keywords (case-insensitive)."""
    title = norm_text(p.get("title"))
    abstract = norm_text(p.get("abstract"))
    kws = p.get("keywords") or []
    if isinstance(kws, list):
        kws_join = " ".join(k.strip() for k in kws if k)
    else:
        kws_join = str(kws)
    haystack = f"{title}\n{abstract}\n{kws_join}".lower()
    for t in terms:
        t = norm_text(t)
        if not t:
            continue
        # escape term and do case-insensitive search (substring)
        if re.search(re.escape(t), haystack, flags=re.IGNORECASE):
            return True
    return False


def author_display_name(a: Dict[str, Any]) -> str:
    return (
        a.get("name")
        or a.get("profile_id")
        or a.get("unclaimed_id")
        or (a.get("emails") or [None])[0]
        or "Unknown Author"
    )


def collect_matches(payload: Dict[str, Any], terms: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    papers: List[Dict[str, Any]] = payload.get("papers", [])
    authors: List[Dict[str, Any]] = payload.get("authors", [])
    paper_by_id: Dict[str, Dict[str, Any]] = {p["paper_id"]: p for p in papers if p.get("paper_id")}

    # 1) filter papers by terms
    matched_paper_ids: Set[str] = set()
    matched_papers: List[Dict[str, Any]] = []
    for p in papers:
        if paper_matches(p, terms):
            matched_papers.append(p)
            matched_paper_ids.add(p["paper_id"])

    # 2) build per-author matched paper lists
    matched_authors: List[Dict[str, Any]] = []
    for a in authors:
        authored = a.get("papers", []) or []
        # Keep only those authored papers that are in matched_paper_ids
        mp = []
        for ap in authored:
            pid = ap.get("paper_id")
            if pid in matched_paper_ids:
                # merge with canonical paper info (to get forum link etc.)
                full = paper_by_id.get(pid, {})
                mp.append({
                    "paper_id": pid,
                    "title": ap.get("title") or full.get("title"),
                    "venue": ap.get("venue") or full.get("venue"),
                    "forum": full.get("forum"),
                })
        if not mp:
            continue

        # Build author view model
        entry = {
            "name": author_display_name(a),
            "profile_url": a.get("profile_url", None),
            "paper_count": len(mp),
            "papers": sorted(mp, key=lambda x: (x["title"] or "").lower()),
            "history": a.get("history") or [],
        }
        if entry['profile_url'] is None and a.get("unclaimed_id"):
            entry['profile_url'] = f"https://openreview.net/profile?id={a['unclaimed_id']}"
        matched_authors.append(entry)

    # 3) sort authors by number of matched papers desc, then by name
    matched_authors.sort(key=lambda x: (-x["paper_count"], x["name"].lower()))
    return matched_papers, matched_authors


def render_html(template_dir: str,
                template_name: str,
                output_path: str,
                *,
                terms: List[str],
                matched_papers: List[Dict[str, Any]],
                matched_authors: List[Dict[str, Any]]) -> None:
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        extensions=['jinja2.ext.loopcontrols']
    )
    template = env.get_template(template_name)
    html = template.render(
        terms=terms,
        matched_papers_count=len(matched_papers),
        matched_authors_count=len(matched_authors),
        authors=matched_authors
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Build a topic page of authors from OpenReview payload JSON.")
    parser.add_argument("--input", required=True, help="Path to fetch OpenReview content")
    parser.add_argument("--terms", required=True, help="Search terms (OR semantics).")
    parser.add_argument("--template", default="authors_by_topic.html.j2", help="Jinja template filename.")
    parser.add_argument("--template-dir", default=".", help="Directory containing the Jinja template.")
    parser.add_argument("--out", default="authors_topic.html", help="Output HTML file.")
    parser.add_argument("--limit", type=int, default=None, help="Show only top N authors.")
    parser.add_argument("--min-papers", type=int, default=1, help="Only include authors with at least N matched papers.")
    args = parser.parse_args()

    payload = load_payload(args.input)
    args.terms = args.terms.split(",")
    matched_papers, matched_authors = collect_matches(payload, args.terms)

    # apply min-papers filter and limit
    filtered = [a for a in matched_authors if a["paper_count"] >= args.min_papers]
    if args.limit is not None:
        filtered = filtered[: args.limit]

    # render
    render_html(
        template_dir=args.template_dir,
        template_name=args.template,
        output_path=args.out,
        terms=args.terms,
        matched_papers=matched_papers,
        matched_authors=filtered
    )

    print(f"Wrote {len(filtered)} authors (of {len(matched_authors)}) "
          f"and {len(matched_papers)} matched papers to {args.out}")


if __name__ == "__main__":
    main()
