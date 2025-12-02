from __future__ import annotations

from typing import List, Dict
from pydantic import BaseModel
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time


FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/c238/Automotive/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://www.dealnews.com/c196/Home-Garden/?rss=1",
]


def extract(html_snippet: str) -> str:
    """
    Use BeautifulSoup to clean up this HTML snippet and extract useful text.
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet

    return result.replace("\n", " ")


class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed.
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict.
        """
        self.title = entry["title"]
        self.summary = extract(entry["summary"])
        self.url = entry["links"][0]["href"]

        resp = requests.get(self.url)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")
        content = soup.find("div", class_="content-section").get_text()
        content = content.replace("\nmore", "").replace("\n", " ")

        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""

    def __repr__(self) -> str:
        return f"<{self.title}>"

    def describe(self) -> str:
        """
        Return a longer string to describe this deal for use in calling a model.
        """
        return (
            f"Title: {self.title}\n"
            f"Details: {self.details.strip()}\n"
            f"Features: {self.features.strip()}\n"
            f"URL: {self.url}"
        )

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List["ScrapedDeal"]:
        """
        Retrieve all deals from the selected RSS feeds.
        """
        deals: List[ScrapedDeal] = []
        feed_iter = tqdm(FEEDS) if show_progress else FEEDS
        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                deals.append(cls(entry))
                time.sleep(0.5)
        return deals


class Deal(BaseModel):
    """
    A class to represent a Deal with a summary description.
    """
    product_description: str
    price: float
    url: str


class DealSelection(BaseModel):
    """
    A class to represent a list of Deals.
    """
    deals: List[Deal]


class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered.
    """
    deal: Deal
    estimate: float
    discount: float
