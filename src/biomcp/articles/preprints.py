"""Preprint search functionality for bioRxiv/medRxiv and Europe PMC."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .. import http_client, render
from ..constants import (
    BIORXIV_BASE_URL,
    BIORXIV_DEFAULT_DAYS_BACK,
    BIORXIV_MAX_PAGES,
    BIORXIV_RESULTS_PER_PAGE,
    EUROPE_PMC_BASE_URL,
    EUROPE_PMC_PAGE_SIZE,
    MEDRXIV_BASE_URL,
    SYSTEM_PAGE_SIZE,
)
from ..core import PublicationState
from .search import PubmedRequest, ResultItem, SearchResponse

logger = logging.getLogger(__name__)


class BiorxivRequest(BaseModel):
    """Request parameters for bioRxiv/medRxiv API."""

    query: str
    interval: str = Field(
        default="", description="Date interval in YYYY-MM-DD/YYYY-MM-DD format"
    )
    cursor: int = Field(default=0, description="Starting position")


class BiorxivResult(BaseModel):
    """Individual result from bioRxiv/medRxiv."""

    doi: str | None = None
    title: str | None = None
    authors: str | None = None
    author_corresponding: str | None = None
    author_corresponding_institution: str | None = None
    date: str | None = None
    version: int | None = None
    type: str | None = None
    license: str | None = None
    category: str | None = None
    jats: str | None = None
    abstract: str | None = None
    published: str | None = None
    server: str | None = None

    @property
    def doi_url(self) -> str | None:
        """Generate DOI URL if DOI exists."""
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return None

    @property
    def preprint_url(self) -> str | None:
        """Generate preprint URL if DOI exists."""
        if self.doi:
            if self.server == "medrxiv":
                return f"https://www.medrxiv.org/content/{self.doi}"
            else:
                return f"https://www.biorxiv.org/content/{self.doi}"
        return None


class BiorxivResponse(BaseModel):
    """Response from bioRxiv/medRxiv API."""

    messages: list[Any] = Field(default_factory=list)
    cursor: int = 0
    count: int = 0
    total: int = 0
    results: list[BiorxivResult] = Field(default_factory=list)


class EuropePMCRequest(BaseModel):
    """Request parameters for Europe PMC API."""

    query: str
    pageSize: int = Field(default=EUROPE_PMC_PAGE_SIZE, description="Results per page")
    page: int = Field(default=1, description="Page number")
    format: str = Field(default="json", description="Response format")


class EuropePMCResult(BaseModel):
    """Individual result from Europe PMC."""

    id: str | None = None
    source: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    title: str | None = None
    authorString: str | None = None
    journalTitle: str | None = None
    pubYear: str | None = None
    pubMonth: str | None = None
    pubDay: str | None = None
    abstractText: str | None = None
    keywordsList: list[str] | None = None
    journalInfo: dict[str, Any] | None = None

    @property
    def date(self) -> str | None:
        """Format publication date."""
        if self.pubYear:
            year = self.pubYear
            month = self.pubMonth or "01"
            day = self.pubDay or "01"
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return None

    @property
    def authors(self) -> list[str] | None:
        """Parse authors from authorString."""
        if self.authorString:
            return [author.strip() for author in self.authorString.split(",")]
        return None

    @property
    def doi_url(self) -> str | None:
        """Generate DOI URL if DOI exists."""
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return None

    @property
    def pubmed_url(self) -> str | None:
        """Generate PubMed URL if PMID exists."""
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return None

    @property
    def pmc_url(self) -> str | None:
        """Generate PMC URL if PMCID exists."""
        if self.pmcid:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{self.pmcid}/"
        return None


class EuropePMCResponse(BaseModel):
    """Response from Europe PMC API."""

    request: dict[str, Any] | None = None
    version: str | None = None
    timestamp: str | None = None
    hitCount: int = 0
    pageSize: int = 0
    requestId: str | None = None
    resultList: dict[str, Any] | None = None
    results: list[EuropePMCResult] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        if self.resultList and "result" in self.resultList:
            self.results = [
                EuropePMCResult(**item) for item in self.resultList["result"]
            ]


def _convert_to_result_item(biorxiv_result: BiorxivResult) -> ResultItem:
    """Convert BiorxivResult to ResultItem."""
    return ResultItem(
        pmid=None,
        pmcid=None,
        title=biorxiv_result.title,
        journal=None,
        authors=biorxiv_result.authors.split(",") if biorxiv_result.authors else None,
        date=biorxiv_result.date,
        doi=biorxiv_result.doi,
        abstract=biorxiv_result.abstract,
        publication_state=PublicationState.PREPRINT,
        source=biorxiv_result.server or "preprint",
    )


def _convert_europepmc_to_result_item(europepmc_result: EuropePMCResult) -> ResultItem:
    """Convert EuropePMCResult to ResultItem."""
    return ResultItem(
        pmid=int(europepmc_result.pmid) if europepmc_result.pmid else None,
        pmcid=europepmc_result.pmcid,
        title=europepmc_result.title,
        journal=europepmc_result.journalTitle,
        authors=europepmc_result.authors,
        date=europepmc_result.date,
        doi=europepmc_result.doi,
        abstract=europepmc_result.abstractText,
        publication_state=PublicationState.PREPRINT,
        source="Europe PMC",
    )


def _build_biorxiv_query(request: PubmedRequest) -> str:
    """Build query string for bioRxiv/medRxiv."""
    query_parts = []

    # Add keywords
    if request.keywords:
        query_parts.extend(request.keywords)

    # Add diseases
    if request.diseases:
        query_parts.extend(request.diseases)

    # Add genes
    if request.genes:
        query_parts.extend(request.genes)

    # Add chemicals
    if request.chemicals:
        query_parts.extend(request.chemicals)

    # Add variants
    if request.variants:
        query_parts.extend(request.variants)

    return " ".join(query_parts)


def _build_europepmc_query(request: PubmedRequest) -> str:
    """Build query string for Europe PMC."""
    query_parts = []

    # Add keywords
    if request.keywords:
        query_parts.extend(request.keywords)

    # Add diseases with proper formatting
    if request.diseases:
        for disease in request.diseases:
            query_parts.append(f'"{disease}"')

    # Add genes
    if request.genes:
        query_parts.extend(request.genes)

    # Add chemicals
    if request.chemicals:
        query_parts.extend(request.chemicals)

    # Add variants
    if request.variants:
        query_parts.extend(request.variants)

    return " AND ".join(query_parts)


def _get_date_interval(days_back: int = BIORXIV_DEFAULT_DAYS_BACK) -> str:
    """Generate date interval for bioRxiv search."""
    from datetime import timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"


async def _search_biorxiv(
    request: PubmedRequest, max_results: int = SYSTEM_PAGE_SIZE
) -> list[ResultItem]:
    """Search bioRxiv for preprints."""
    query = _build_biorxiv_query(request)
    if not query:
        return []

    results = []
    cursor = 0
    pages_searched = 0

    while len(results) < max_results and pages_searched < BIORXIV_MAX_PAGES:
        biorxiv_request = BiorxivRequest(
            query=query,
            interval=_get_date_interval(),
            cursor=cursor,
        )

        response, error = await http_client.request_api(
            url=BIORXIV_BASE_URL,
            request=biorxiv_request,
            response_model_type=BiorxivResponse,
            domain="preprint",
        )

        if error or not response:
            logger.warning(f"bioRxiv search failed: {error}")
            break

        if not response.results:
            break

        # Convert results
        for result in response.results:
            if len(results) >= max_results:
                break
            results.append(_convert_to_result_item(result))

        cursor += BIORXIV_RESULTS_PER_PAGE
        pages_searched += 1

    return results


async def _search_medrxiv(
    request: PubmedRequest, max_results: int = SYSTEM_PAGE_SIZE
) -> list[ResultItem]:
    """Search medRxiv for preprints."""
    query = _build_biorxiv_query(request)
    if not query:
        return []

    results = []
    cursor = 0
    pages_searched = 0

    while len(results) < max_results and pages_searched < BIORXIV_MAX_PAGES:
        biorxiv_request = BiorxivRequest(
            query=query,
            interval=_get_date_interval(),
            cursor=cursor,
        )

        response, error = await http_client.request_api(
            url=MEDRXIV_BASE_URL,
            request=biorxiv_request,
            response_model_type=BiorxivResponse,
            domain="preprint",
        )

        if error or not response:
            logger.warning(f"medRxiv search failed: {error}")
            break

        if not response.results:
            break

        # Convert results
        for result in response.results:
            if len(results) >= max_results:
                break
            results.append(_convert_to_result_item(result))

        cursor += BIORXIV_RESULTS_PER_PAGE
        pages_searched += 1

    return results


async def _search_europepmc(
    request: PubmedRequest, max_results: int = SYSTEM_PAGE_SIZE
) -> list[ResultItem]:
    """Search Europe PMC for preprints."""
    query = _build_europepmc_query(request)
    if not query:
        return []

    results = []
    page = 1
    max_pages = (max_results // EUROPE_PMC_PAGE_SIZE) + 1

    while len(results) < max_results and page <= max_pages:
        europepmc_request = EuropePMCRequest(
            query=query,
            pageSize=EUROPE_PMC_PAGE_SIZE,
            page=page,
        )

        response, error = await http_client.request_api(
            url=EUROPE_PMC_BASE_URL,
            request=europepmc_request,
            response_model_type=EuropePMCResponse,
            domain="preprint",
        )

        if error or not response:
            logger.warning(f"Europe PMC search failed: {error}")
            break

        if not response.results:
            break

        # Convert results
        for result in response.results:
            if len(results) >= max_results:
                break
            results.append(_convert_europepmc_to_result_item(result))

        page += 1

    return results


async def search_preprints(
    request: PubmedRequest, output_json: bool = False
) -> str:
    """Search for preprints across bioRxiv, medRxiv, and Europe PMC."""
    # Run all searches in parallel
    tasks = [
        _search_biorxiv(request),
        _search_medrxiv(request),
        _search_europepmc(request),
    ]

    results_lists = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results
    all_results = []
    for results in results_lists:
        if isinstance(results, list):
            all_results.extend(results)
        elif isinstance(results, Exception):
            logger.warning(f"Preprint search failed: {results}")

    # Remove duplicates based on DOI
    unique_results = []
    seen_dois = set()
    for result in all_results:
        if result.doi and result.doi in seen_dois:
            continue
        if result.doi:
            seen_dois.add(result.doi)
        unique_results.append(result)

    # Sort by date (newest first)
    unique_results.sort(
        key=lambda x: x.date or "0000-00-00", reverse=True
    )

    # Convert to dict format for rendering
    data = [
        result.model_dump(mode="json", exclude_none=True)
        for result in unique_results
    ]

    if output_json:
        return json.dumps(data, indent=2)
    else:
        return render.to_markdown(data)