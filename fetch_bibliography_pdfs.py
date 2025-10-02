"""Fetch PDFs for references listed in a bibliography.

This module provides a command line interface that accepts a block of
bibliography text, attempts to discover DOIs for each entry and downloads
available PDF files.  It relies on Crossref to resolve DOIs when they are
not explicitly included in the bibliography.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import textwrap
from typing import List, Optional, Sequence, Tuple

import requests

# DOI regular expression adapted from the Crossref recommendation:
# https://www.crossref.org/blog/dois-and-matching-regular-expressions/
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
USER_AGENT_TEMPLATE = (
    "BibliographyPDFDownloader/1.0 (+https://example.com; mailto:{email})"
)
DEFAULT_USER_AGENT = "BibliographyPDFDownloader/1.0"


def split_references(text: str) -> List[str]:
    """Split a block of bibliography text into individual references.

    The function first attempts to split on blank lines.  If the text is a
    numbered list without blank line separators, it falls back to detecting
    numbering prefixes.
    """

    text = text.strip()
    if not text:
        return []

    # Try blank-line separation first.
    references = [segment.strip() for segment in re.split(r"\n\s*\n", text)]
    references = [ref for ref in references if ref]

    if len(references) > 1:
        return references

    # Fallback: split on numbering patterns like "1. " or "[1] "
    numbered_pattern = re.compile(r"\n(?=\s*(?:\[\d+\]|\d+)[\.)]\s)")
    numbered_refs = [segment.strip() for segment in numbered_pattern.split(text)]
    numbered_refs = [ref for ref in numbered_refs if ref]

    return numbered_refs if len(numbered_refs) > 1 else references


def extract_doi(reference: str) -> Optional[str]:
    """Return the first DOI present in *reference*, if any."""

    match = DOI_PATTERN.search(reference)
    if match:
        doi = match.group(0)
        # Strip trailing punctuation that is common in bibliographies.
        return doi.rstrip(".,;)]")
    return None


def lookup_doi(
    reference: str, session: requests.Session, user_agent: str
) -> Optional[str]:
    """Attempt to resolve a DOI using the Crossref works API."""

    url = "https://api.crossref.org/works"
    params = {"query.bibliographic": reference, "rows": 1}
    headers = {"User-Agent": user_agent}

    response = session.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    items = data.get("message", {}).get("items", [])
    if not items:
        return None
    return items[0].get("DOI")


def find_pdf_link(
    doi: str, session: requests.Session, user_agent: str
) -> Optional[str]:
    """Try to identify a direct PDF link for *doi* using Crossref metadata."""

    url = f"https://api.crossref.org/works/{requests.utils.quote(doi)}"
    headers = {"User-Agent": user_agent}
    response = session.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        return None

    data = response.json().get("message", {})
    for link in data.get("link", []):
        content_type = link.get("content-type", "").lower()
        if "pdf" in content_type and link.get("URL"):
            return link["URL"]

    return None


def sanitize_filename(text: str, max_length: int = 120) -> str:
    """Return a filesystem-friendly version of *text*."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return sanitized[:max_length] or "reference"


def reference_title(reference: str) -> str:
    """Extract an approximate title from a reference for file naming."""

    # Attempt to capture text between first period and following period as title.
    parts = reference.split(".")
    if len(parts) > 2:
        # Skip potential author list (first part) and return the next piece.
        candidate = parts[1].strip()
        if candidate:
            return candidate
    # Fallback to truncated reference.
    return reference[:80]


def download_pdf(
    doi: str,
    destination: pathlib.Path,
    session: requests.Session,
    user_agent: str,
) -> Tuple[bool, Optional[pathlib.Path], Optional[str]]:
    """Download a PDF for *doi* and store it at *destination* directory."""

    headers = {"User-Agent": user_agent, "Accept": "application/pdf"}
    pdf_url = find_pdf_link(doi, session, user_agent)
    download_urls = [pdf_url] if pdf_url else []
    download_urls.append(f"https://doi.org/{doi}")

    for url in download_urls:
        if not url:
            continue
        try:
            response = session.get(url, headers=headers, stream=True, timeout=60)
            if "pdf" not in response.headers.get("Content-Type", "").lower():
                # Some publishers redirect to HTML; skip such responses.
                continue

            filename = sanitize_filename(doi)
            target_path = destination / f"{filename}.pdf"
            with target_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            return True, target_path, None
        except requests.RequestException as exc:
            last_error = str(exc)
            continue
    else:
        last_error = "No PDF response received"

    return False, None, last_error


def process_references(
    references: Sequence[str],
    output_dir: pathlib.Path,
    session: requests.Session,
    user_agent: str,
    lookup_missing_doi: bool,
) -> List[Tuple[str, Optional[str], Optional[pathlib.Path], Optional[str]]]:
    """Process each reference and attempt to download its PDF."""

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, reference in enumerate(references, start=1):
        print(f"[{index}/{len(references)}] Processing reference…")
        doi = extract_doi(reference)
        resolved_doi = doi
        error = None

        if not doi and lookup_missing_doi:
            try:
                resolved_doi = lookup_doi(reference, session, user_agent)
            except requests.RequestException as exc:
                error = f"Crossref lookup failed: {exc}"

        pdf_path = None
        if resolved_doi:
            try:
                success, pdf_path, download_error = download_pdf(
                    resolved_doi, output_dir, session, user_agent
                )
                if not success:
                    error = download_error or "Unknown download error"
            except requests.RequestException as exc:
                error = f"Download failed: {exc}"
        else:
            if not error:
                error = "No DOI available"

        results.append((reference, resolved_doi, pdf_path, error))

    return results


def format_summary(
    results: Sequence[Tuple[str, Optional[str], Optional[pathlib.Path], Optional[str]]]
) -> str:
    """Return a human-readable summary of the processing results."""

    lines = []
    for reference, doi, pdf_path, error in results:
        title = reference_title(reference)
        if pdf_path:
            lines.append(f"✔ {title} -> {pdf_path}")
        else:
            detail = error or "Unknown error"
            if doi:
                lines.append(f"✖ {title} (DOI: {doi}) — {detail}")
            else:
                lines.append(f"✖ {title} — {detail}")
    return "\n".join(lines)


def build_user_agent(email: Optional[str]) -> str:
    if email:
        return USER_AGENT_TEMPLATE.format(email=email)
    return DEFAULT_USER_AGENT


def read_input_text(path: Optional[str]) -> str:
    if path:
        return pathlib.Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download PDFs for citations in a bibliography block.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples
            --------
            Fetch PDFs for references stored in a text file:
              python fetch_bibliography_pdfs.py --input refs.txt --outdir downloads \
                  --email your.name@example.edu

            Read references from the clipboard or shell pipeline:
              pbpaste | python fetch_bibliography_pdfs.py
            """
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="FILE",
        help="Path to a text file containing the bibliography (defaults to stdin)",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default="downloads",
        help="Directory where downloaded PDFs will be stored",
    )
    parser.add_argument(
        "--email",
        help=(
            "Contact email used for the Crossref polite pool User-Agent."
            " Providing this increases the chance of successful DOI lookups."
        ),
    )
    parser.add_argument(
        "--no-lookup",
        action="store_true",
        help="Do not query Crossref when a reference lacks an explicit DOI",
    )

    args = parser.parse_args(argv)

    text = read_input_text(args.input)
    references = split_references(text)
    if not references:
        print("No references found in the provided input.")
        return 1

    output_dir = pathlib.Path(args.outdir)
    user_agent = build_user_agent(args.email)

    session = requests.Session()
    results = process_references(
        references,
        output_dir,
        session,
        user_agent,
        lookup_missing_doi=not args.no_lookup,
    )

    print()
    print(format_summary(results))

    failures = [result for result in results if result[2] is None]
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
