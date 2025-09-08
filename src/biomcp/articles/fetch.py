"""Article fetching functionality."""

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field

from .. import http_client, render
from ..constants import PUBTATOR3_FULLTEXT_URL


class FetchRequest(BaseModel):
    pmids: list[int] = Field(description="List of PMIDs to fetch")
    full: bool = Field(default=False, description="Fetch full text")


class ArticleData(BaseModel):
    pmid: int | None = None
    title: str | None = None
    abstract: str | None = None
    full_text: str | None = None
    authors: list[str] | None = None
    journal: str | None = None
    date: str | None = None
    doi: str | None = None


class FetchResponse(BaseModel):
    articles: list[ArticleData] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def get_abstract(self, pmid: int) -> str | None:
        """Get abstract for a specific PMID."""
        for article in self.articles:
            if article.pmid == pmid:
                return article.abstract
        return None


async def call_pubtator_api(
    pmids: list[int], full: bool = False
) -> tuple[FetchResponse | None, str | None]:
    """Call PubTator API to fetch article data."""
    if not pmids:
        return None, "No PMIDs provided"

    request = FetchRequest(pmids=pmids, full=full)

    response, error = await http_client.request_api(
        url=PUBTATOR3_FULLTEXT_URL,
        request=request,
        response_model_type=FetchResponse,
        domain="article",
    )

    if error:
        return None, f"Error {error.code}: {error.message}"

    return response, None


async def fetch_articles(
    identifiers: list[str | int], full: bool = False, output_json: bool = True
) -> str:
    """Fetch articles by PMID or DOI."""
    pmids = []
    dois = []

    # Separate PMIDs and DOIs
    for identifier in identifiers:
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            pmids.append(int(identifier))
        else:
            dois.append(identifier)

    results = []

    # Fetch by PMID
    if pmids:
        response, error = await call_pubtator_api(pmids, full)
        if response:
            for article in response.articles:
                results.append(article.model_dump(exclude_none=True))
        elif error:
            results.append({"error": error})

    # For DOIs, we would need additional API calls
    # This is a placeholder for future implementation
    if dois:
        for doi in dois:
            results.append({
                "doi": doi,
                "error": "DOI fetching not yet implemented"
            })

    if output_json:
        return json.dumps(results, indent=2)
    else:
        return render.to_markdown(results)


def is_pmid(identifier: str) -> bool:
    """Check if identifier is a PMID."""
    return identifier.isdigit()


def is_doi(identifier: str) -> bool:
    """Check if identifier is a DOI."""
    return identifier.startswith("10.") and "/" in identifier