"""Download sample documents from the Philippine Official Gazette."""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
METADATA_FILE = Path(__file__).parent.parent / "data" / "metadata.json"

# Curated list of Official Gazette pages (laws, executive orders, proclamations)
SAMPLE_URLS = [
    {
        "url": "https://www.officialgazette.gov.ph/2012/09/12/republic-act-no-10175/",
        "title": "Republic Act No. 10175 - Cybercrime Prevention Act of 2012",
        "type": "law",
        "date": "2012-09-12",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2013/05/15/republic-act-no-10591/",
        "title": "Republic Act No. 10591 - Comprehensive Firearms and Ammunition Regulation Act",
        "type": "law",
        "date": "2013-05-15",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2012/12/19/republic-act-no-10354/",
        "title": "Republic Act No. 10354 - Responsible Parenthood and Reproductive Health Act",
        "type": "law",
        "date": "2012-12-19",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2013/03/29/republic-act-no-10533/",
        "title": "Republic Act No. 10533 - Enhanced Basic Education Act (K-12)",
        "type": "law",
        "date": "2013-03-29",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2019/02/14/republic-act-no-11232/",
        "title": "Republic Act No. 11232 - Revised Corporation Code of the Philippines",
        "type": "law",
        "date": "2019-02-14",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2018/01/05/republic-act-no-10963/",
        "title": "Republic Act No. 10963 - TRAIN Law (Tax Reform for Acceleration and Inclusion)",
        "type": "law",
        "date": "2018-01-05",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2019/07/26/republic-act-no-11313/",
        "title": "Republic Act No. 11313 - Safe Spaces Act (Bawal Bastos Law)",
        "type": "law",
        "date": "2019-07-26",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2013/02/15/republic-act-no-10368/",
        "title": "Republic Act No. 10368 - Human Rights Victims Reparation and Recognition Act",
        "type": "law",
        "date": "2013-02-15",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2019/01/11/republic-act-no-11223/",
        "title": "Republic Act No. 11223 - Universal Health Care Act",
        "type": "law",
        "date": "2019-01-11",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2022/12/06/republic-act-no-11934/",
        "title": "Republic Act No. 11934 - SIM Registration Act",
        "type": "law",
        "date": "2022-12-06",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2012/08/22/republic-act-no-10173/",
        "title": "Republic Act No. 10173 - Data Privacy Act of 2012",
        "type": "law",
        "date": "2012-08-22",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2017/05/23/republic-act-no-10928/",
        "title": "Republic Act No. 10928 - Free Irrigation Service Act",
        "type": "law",
        "date": "2017-05-23",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2016/07/21/republic-act-no-10844/",
        "title": "Republic Act No. 10844 - Department of Information and Communications Technology Act",
        "type": "law",
        "date": "2016-07-21",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2018/07/26/republic-act-no-11058/",
        "title": "Republic Act No. 11058 - Occupational Safety and Health Standards Act",
        "type": "law",
        "date": "2018-07-26",
    },
    {
        "url": "https://www.officialgazette.gov.ph/2019/08/08/republic-act-no-11332/",
        "title": "Republic Act No. 11332 - Mandatory Reporting of Notifiable Diseases Act",
        "type": "law",
        "date": "2019-08-08",
    },
]

HEADERS = {
    "User-Agent": "PH-Politician-RAG/0.1 (Research Project; github.com)"
}


def extract_text(url: str) -> str | None:
    """Fetch a page and extract its main text content."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Official Gazette uses .entry-content for the main body
    content = soup.find("div", class_="entry-content")
    if not content:
        # Fallback: try article tag or main content area
        content = soup.find("article") or soup.find("main")
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

    print(f"Downloading {len(SAMPLE_URLS)} sample documents from Official Gazette...\n")

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
                "source": "Official Gazette",
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
