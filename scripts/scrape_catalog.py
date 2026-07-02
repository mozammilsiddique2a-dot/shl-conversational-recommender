import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from requests import Session
from requests.exceptions import RequestException, Timeout


CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "catalog.json"
TIMEOUT_SECONDS = 20
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SHLAssessmentRecommender/1.0; "
        "+https://github.com/openai/codex)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


logger = logging.getLogger("scrape_catalog")
GENERIC_LINK_TEXT = {
    "",
    "learn more",
    "read guide",
    "read paper",
    "view all shl products",
    "view all shl solutions",
}


@dataclass(frozen=True)
class CatalogItem:
    name: str
    url: str
    test_type: str
    description: str
    category: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type,
            "description": self.description,
            "category": self.category,
        }


def clean_text(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def make_session() -> Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_html(session: Session, url: str, timeout: int) -> str | None:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except Timeout:
        logger.warning("Timed out fetching %s after %s seconds", url, timeout)
        return None
    except RequestException as exc:
        logger.warning("Could not fetch %s: %s", url, exc)
        return None
    return response.text


def heading_before(node: Tag) -> str | None:
    for previous in node.find_all_previous(["h1", "h2", "h3", "h4"], limit=4):
        text = clean_text(previous.get_text(" ", strip=True))
        if text and text.lower() != "outdated browser detected":
            return text
    return None


def absolute_url(href: str, base_url: str) -> str:
    return urljoin(base_url, href)


def is_generic_catalog_url(url: str, base_url: str) -> bool:
    normalized = url.rstrip("/").lower()
    generic = base_url.rstrip("/").lower()
    return normalized == generic or normalized in {
        "https://www.shl.com/products",
        "https://www.shl.com/products/assessments",
        "https://www.shl.com/solutions/products/product-catalog",
    }


def candidate_urls(node: Tag, base_url: str) -> list[str]:
    urls: list[str] = []
    for attr in ("data-url", "data-href", "data-link", "href"):
        value = node.get(attr)
        if isinstance(value, str) and value.strip():
            urls.append(absolute_url(value.strip(), base_url))
    for child in node.select("[data-url], [data-href], [data-link], a[href]"):
        for attr in ("data-url", "data-href", "data-link", "href"):
            value = child.get(attr)
            if isinstance(value, str) and value.strip():
                urls.append(absolute_url(value.strip(), base_url))
    return urls


def product_url_for_node(node: Tag, base_url: str) -> str:
    urls = candidate_urls(node, base_url)
    if not urls:
        return base_url

    productish = [
        url
        for url in urls
        if "shl.com" in url.lower()
        and not is_generic_catalog_url(url, base_url)
        and not url.endswith("#")
    ]
    if productish:
        return productish[0]

    non_generic = [url for url in urls if not is_generic_catalog_url(url, base_url)]
    if non_generic:
        return non_generic[0]
    return urls[0]


def is_product_specific_url(url: str, base_url: str) -> bool:
    lowered = url.lower().rstrip("/")
    if is_generic_catalog_url(url, base_url):
        return False
    return (
        "shl.com/products/assessments/" in lowered
        and lowered not in {
            "https://www.shl.com/products/assessments/behavioral-assessments",
            "https://www.shl.com/products/assessments/personality-assessment",
            "https://www.shl.com/products/assessments/skills-and-simulations",
        }
    )


def looks_like_individual_solution(text: str) -> bool:
    lowered = text.lower()
    return "individual test solution" in lowered or "individual test solutions" in lowered


def normalize_test_type(value: str) -> str:
    value = clean_text(value)
    if not value:
        return "Unknown"
    lowered = value.lower()
    aliases = {
        "a": "Ability & Aptitude",
        "b": "Biodata & Situational Judgement",
        "c": "Competencies",
        "d": "Development & 360",
        "e": "Assessment Exercise",
        "k": "Knowledge & Skills",
        "p": "Personality & Behavior",
        "s": "Simulations",
    }
    if lowered in aliases:
        return aliases[lowered]
    return value


def detail_description(session: Session, url: str, timeout: int) -> str | None:
    html = fetch_html(session, url, timeout)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.select_one("meta[name='description'], meta[property='og:description']")
    if meta and meta.get("content"):
        return clean_text(meta["content"])

    main = soup.select_one("main") or soup.body
    if not main:
        return None
    for paragraph in main.select("p"):
        text = clean_text(paragraph.get_text(" ", strip=True))
        if len(text) >= 80:
            return text
    return None


def parse_table(table: Tag, base_url: str) -> Iterable[CatalogItem]:
    headers = [clean_text(cell.get_text(" ", strip=True)).lower() for cell in table.select("thead th")]
    category = heading_before(table)

    for row in table.select("tbody tr"):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
        link = row.select_one("a[href]")
        if not link:
            continue

        name = clean_text(link.get_text(" ", strip=True)) or values[0]
        url = product_url_for_node(row, base_url)
        if not is_product_specific_url(url, base_url):
            logger.info("Skipping %s because no product-specific URL was found", name)
            continue
        row_text = " ".join(values)

        if category and not looks_like_individual_solution(category + " " + row_text):
            continue

        test_type = "Unknown"
        for index, header in enumerate(headers):
            if index >= len(values):
                continue
            if "test type" in header or "type" == header:
                test_type = normalize_test_type(values[index])
                break

        if test_type == "Unknown" and len(values) > 1:
            test_type = normalize_test_type(values[1])

        description = clean_text(row.get("data-description")) or f"SHL Individual Test Solution: {name}."
        yield CatalogItem(name=name, url=url, test_type=test_type, description=description, category=category)


def parse_cards(soup: BeautifulSoup, base_url: str) -> Iterable[CatalogItem]:
    selectors = [
        ".product-card",
        ".card",
        "article",
        "li",
    ]
    seen_nodes: set[int] = set()
    for selector in selectors:
        for card in soup.select(selector):
            if id(card) in seen_nodes:
                continue
            seen_nodes.add(id(card))
            link = card.select_one("a[href]")
            if not link:
                continue
            name = clean_text(link.get_text(" ", strip=True))
            if not name or len(name) < 3:
                continue

            card_text = clean_text(card.get_text(" ", strip=True))
            category = heading_before(card)
            if category and not looks_like_individual_solution(category + " " + card_text):
                continue
            if not category and "individual test" not in card_text.lower():
                continue

            test_type = "Unknown"
            type_match = re.search(r"(?:test\s*type|type)\s*:?\s*([A-Za-z &/]+)", card_text, re.IGNORECASE)
            if type_match:
                test_type = normalize_test_type(type_match.group(1))

            description = card_text.replace(name, "", 1).strip(" -|")
            if not description:
                description = f"SHL Individual Test Solution: {name}."

            url = product_url_for_node(card, base_url)
            if not is_product_specific_url(url, base_url):
                logger.info("Skipping %s because no product-specific URL was found", name)
                continue

            yield CatalogItem(
                name=name,
                url=url,
                test_type=test_type,
                description=description,
                category=category,
            )


def parse_current_products_page(soup: BeautifulSoup, base_url: str) -> Iterable[CatalogItem]:
    for link in soup.select("a[href]"):
        name = clean_text(link.get_text(" ", strip=True))
        url = absolute_url(link.get("href", ""), base_url)
        if name.lower() in GENERIC_LINK_TEXT or not is_product_specific_url(url, base_url):
            continue

        container = link.find_parent(["article", "section", "div", "li"]) or link
        description = clean_text(container.get_text(" ", strip=True)).replace(name, "", 1).strip(" -|")
        if not description:
            description = f"SHL assessment product page: {name}."

        path = url.lower()
        if "personality" in path:
            test_type = "Personality"
        elif "cognitive" in path:
            test_type = "Cognitive Ability"
        elif "skills-and-simulations" in path:
            test_type = "Knowledge and Skills"
        elif "behavioral" in path or "judgement" in path or "judgment" in path:
            test_type = "Behavioral, Situational Judgement"
        else:
            test_type = "Assessment"

        yield CatalogItem(
            name=name,
            url=url,
            test_type=test_type,
            description=description,
            category=heading_before(link),
        )


def parse_catalog(html: str, base_url: str) -> list[CatalogItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[CatalogItem] = []

    for table in soup.select("table"):
        items.extend(parse_table(table, base_url))

    if not items:
        logger.info("No catalog table entries found; falling back to card/list parsing")
        items.extend(parse_cards(soup, base_url))
    if not items:
        logger.info("No Individual Test Solutions section found; falling back to current SHL products page parsing")
        items.extend(parse_current_products_page(soup, base_url))

    return dedupe(items)


def enrich_descriptions(items: list[CatalogItem], session: Session, timeout: int) -> list[CatalogItem]:
    enriched: list[CatalogItem] = []
    for item in items:
        if item.description and not (
            item.description.startswith("SHL Individual Test Solution:")
            or item.description.startswith("SHL assessment product page:")
        ):
            enriched.append(item)
            continue
        description = detail_description(session, item.url, timeout)
        if not description:
            logger.info("Using fallback description for %s", item.name)
            enriched.append(item)
            continue
        enriched.append(
            CatalogItem(
                name=item.name,
                url=item.url,
                test_type=item.test_type,
                description=description,
                category=item.category,
            )
        )
    return enriched


def dedupe(items: Iterable[CatalogItem]) -> list[CatalogItem]:
    item_list = list(items)
    unique: dict[str, CatalogItem] = {}
    for item in item_list:
        key = item.url.rstrip("/").lower()
        if key not in unique:
            unique[key] = item
    removed = len(item_list) - len(unique)
    if removed > 0:
        logger.info("Removed %s duplicate catalog entries", removed)
    return list(unique.values())


def scrape_catalog(url: str = CATALOG_URL, timeout: int = TIMEOUT_SECONDS, enrich: bool = True) -> list[dict]:
    session = make_session()
    html = fetch_html(session, url, timeout)
    if not html:
        logger.warning("No catalog data scraped because the catalog page could not be fetched")
        return []

    items = parse_catalog(html, url)
    if not items:
        logger.warning("No Individual Test Solutions found. The SHL page structure may have changed.")
        return []

    if enrich:
        items = enrich_descriptions(items, session, timeout)

    return [item.to_dict() for item in items]


def write_catalog(items: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape SHL Individual Test Solutions into app/data/catalog.json.")
    parser.add_argument("--url", default=CATALOG_URL, help="SHL catalog URL to scrape.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output JSON path.")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS, help="Request timeout in seconds.")
    parser.add_argument("--no-enrich", action="store_true", help="Skip product detail-page description fetches.")
    parser.add_argument("--allow-empty", action="store_true", help="Write [] if no entries can be scraped.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    items = scrape_catalog(args.url, timeout=args.timeout, enrich=not args.no_enrich)
    if not items:
        logger.warning("No entries scraped from SHL catalog")
        if not args.allow_empty:
            logger.warning("Leaving existing catalog unchanged. Pass --allow-empty to write [].")
            return

    write_catalog(items, args.output)
    logger.warning("Wrote %s catalog entries to %s", len(items), args.output)


if __name__ == "__main__":
    main()
