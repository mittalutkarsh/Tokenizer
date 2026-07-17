#!/usr/bin/env python3
"""
Fetch India Wikipedia pages for English, Hindi, Telugu, Tamil
and convert them to a faithful Markdown corpus.

Keeps visible article content: links, tables, references, navboxes,
categories. Removes only scripts/styles/meta machinery.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urljoin

import regex
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

OUT = Path(__file__).resolve().parent / "corpus"
USER_AGENT = "ERA V5 tokenizer / student build"
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

PAGES = {
    "en":  ("English", "India"),
    "hi":  ("Hindi",   "भारत"),
    "te":  ("Telugu",  "భారతదేశం"),
    "ta":  ("Tamil",   "இந்தியா"),
}


def get(url: str) -> requests.Response:
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(8, 60))


def absolutize_links(soup: BeautifulSoup, lang: str) -> None:
    base = f"https://{lang}.wikipedia.org/wiki/"
    for tag in soup.find_all(["a", "img", "source"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if not value:
            continue
        if value.startswith("//"):
            tag[attr] = "https:" + value
        elif value.startswith("./"):
            tag[attr] = urljoin(base, value[2:])
        elif value.startswith("/"):
            tag[attr] = urljoin(f"https://{lang}.wikipedia.org", value)


def strip_technical_noise(node: BeautifulSoup, soup: BeautifulSoup) -> None:
    for tag in node(["script", "style", "meta"]):
        tag.decompose()
    for tag in node.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        href = tag.get("href") or ""
        if "mw:PageProp/Category" in rel and href:
            tag.replace_with(soup.new_string(f"\nCategory: {href}\n"))
        else:
            tag.decompose()


def normalize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\xa0", " ")
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return markdown.strip() + "\n"


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def build_one(lang: str, name: str, title: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{quote(title)}"
    print(f"  Fetching {name} ({lang}) — {url}")

    res = get(url)
    res.raise_for_status()

    (OUT / f"{lang}.raw.html").write_text(res.text, encoding="utf-8")

    soup = BeautifulSoup(res.text, "lxml")
    body = soup.find("body") or soup
    strip_technical_noise(body, soup)
    absolutize_links(body, lang)
    markdown = normalize_markdown(
        md(str(body), heading_style="ATX", bullets="-", strip=["span"])
    )

    (OUT / f"{lang}.faithful.md").write_text(markdown, encoding="utf-8")
    (OUT / f"{lang}.faithful.txt").write_text(markdown, encoding="utf-8")

    units = faithful_units(markdown)
    meta = {
        "lang": lang,
        "name": name,
        "title": title,
        "source_url": url,
        "chars": len(markdown),
        "faithful_units": units,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / f"{lang}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✓ {name}: {len(markdown):,} chars, {units:,} faithful units")
    return meta


def main() -> int:
    print("=" * 60)
    print("Fetching faithful Markdown corpus from Wikipedia")
    print("Languages: English, Hindi, Telugu, Tamil")
    print("=" * 60)
    for lang, (name, title) in PAGES.items():
        build_one(lang, name, title)
        time.sleep(1)  # polite delay between requests
    print("\nCorpus written to: corpus/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
