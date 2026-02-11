"""Download sample Philippine law documents from lawphil.net."""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
METADATA_FILE = Path(__file__).parent.parent / "data" / "metadata.json"

# Curated list of Philippine laws from lawphil.net
SAMPLE_URLS = [
    {
        "url": "https://lawphil.net/statutes/repacts/ra2012/ra_10175_2012.html",
        "title": "Republic Act No. 10175 - Cybercrime Prevention Act of 2012",
        "type": "law",
        "date": "2012-09-12",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2013/ra_10591_2013.html",
        "title": "Republic Act No. 10591 - Comprehensive Firearms and Ammunition Regulation Act",
        "type": "law",
        "date": "2013-05-15",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2012/ra_10354_2012.html",
        "title": "Republic Act No. 10354 - Responsible Parenthood and Reproductive Health Act",
        "type": "law",
        "date": "2012-12-19",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2013/ra_10533_2013.html",
        "title": "Republic Act No. 10533 - Enhanced Basic Education Act (K-12)",
        "type": "law",
        "date": "2013-03-29",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2019/ra_11232_2019.html",
        "title": "Republic Act No. 11232 - Revised Corporation Code of the Philippines",
        "type": "law",
        "date": "2019-02-14",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2017/ra_10963_2017.html",
        "title": "Republic Act No. 10963 - TRAIN Law (Tax Reform for Acceleration and Inclusion)",
        "type": "law",
        "date": "2018-01-05",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2019/ra_11313_2019.html",
        "title": "Republic Act No. 11313 - Safe Spaces Act (Bawal Bastos Law)",
        "type": "law",
        "date": "2019-07-26",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2013/ra_10368_2013.html",
        "title": "Republic Act No. 10368 - Human Rights Victims Reparation and Recognition Act",
        "type": "law",
        "date": "2013-02-15",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2019/ra_11223_2019.html",
        "title": "Republic Act No. 11223 - Universal Health Care Act",
        "type": "law",
        "date": "2019-01-11",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2022/ra_11934_2022.html",
        "title": "Republic Act No. 11934 - SIM Registration Act",
        "type": "law",
        "date": "2022-12-06",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2012/ra_10173_2012.html",
        "title": "Republic Act No. 10173 - Data Privacy Act of 2012",
        "type": "law",
        "date": "2012-08-22",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2017/ra_10928_2017.html",
        "title": "Republic Act No. 10928 - Free Irrigation Service Act",
        "type": "law",
        "date": "2017-05-23",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2016/ra_10844_2016.html",
        "title": "Republic Act No. 10844 - Department of Information and Communications Technology Act",
        "type": "law",
        "date": "2016-07-21",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2018/ra_11058_2018.html",
        "title": "Republic Act No. 11058 - Occupational Safety and Health Standards Act",
        "type": "law",
        "date": "2018-07-26",
    },
    {
        "url": "https://lawphil.net/statutes/repacts/ra2019/ra_11332_2019.html",
        "title": "Republic Act No. 11332 - Mandatory Reporting of Notifiable Diseases Act",
        "type": "law",
        "date": "2019-08-08",
    },
]


def extract_text(url: str) -> str | None:
    """Fetch a page and extract its main text content."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # lawphil.net uses <body> with the full law text
    content = soup.find("body")
    if not content:
        print(f"  No content found at {url}")
        return None

    # Remove script/style tags
    for tag in content.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = content.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text if len(text) > 100 else None


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []
    downloaded = 0
    skipped = 0

    print(f"Downloading {len(SAMPLE_URLS)} sample documents from lawphil.net...\n")

    for i, entry in enumerate(SAMPLE_URLS):
        print(f"[{i + 1}/{len(SAMPLE_URLS)}] {entry['title']}")

        text = extract_text(entry["url"])
        if not text:
            skipped += 1
            continue

        # Save text file
        filename = f"{entry['title'][:80].replace('/', '-').replace(' ', '_')}.txt"
        filepath = DATA_DIR / filename
        filepath.write_text(text, encoding="utf-8")

        # Track metadata
        metadata.append(
            {
                "filename": filename,
                "title": entry["title"],
                "source": "lawphil.net",
                "url": entry["url"],
                "type": entry["type"],
                "date": entry["date"],
            }
        )

        downloaded += 1
        print(f"  Saved ({len(text):,} chars)")

        # Rate limit
        if i < len(SAMPLE_URLS) - 1:
            time.sleep(1.5)

    # Save metadata
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone! Downloaded {downloaded}, skipped {skipped}")
    print(f"Documents saved to: {DATA_DIR}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()
