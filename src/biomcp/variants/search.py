"""Variant search functionality."""

import json
import logging
from typing import Annotated

from pydantic import BaseModel, Field

from .. import http_client, render
from ..constants import MYVARIANT_QUERY_URL

logger = logging.getLogger(__name__)


class VariantQuery(BaseModel):
    gene: str | None = Field(default=None, description="Gene symbol")
    hgvsp: str | None = Field(default=None, description="Protein notation")
    hgvsc: str | None = Field(default=None, description="cDNA notation")
    rsid: str | None = Field(default=None, description="dbSNP rsID")
    region: str | None = Field(default=None, description="Genomic region")
    significance: str | None = Field(default=None, description="Clinical significance")
    min_frequency: float | None = Field(default=None, description="Minimum frequency")
    max_frequency: float | None = Field(default=None, description="Maximum frequency")
    cadd: float | None = Field(default=None, description="Minimum CADD score")
    polyphen: str | None = Field(default=None, description="PolyPhen prediction")
    sift: str | None = Field(default=None, description="SIFT prediction")
    size: int = Field(default=40, description="Maximum number of results")
    sources: list[str] = Field(default_factory=list, description="Sources to include")


class VariantResult(BaseModel):
    variant_id: str | None = None
    gene: str | None = None
    hgvsp: str | None = None
    hgvsc: str | None = None
    rsid: str | None = None
    chromosome: str | None = None
    position: int | None = None
    ref: str | None = None
    alt: str | None = None
    clinical_significance: str | None = None
    cadd_score: float | None = None
    polyphen_score: float | None = None
    sift_score: float | None = None
    frequency: float | None = None
    sources: list[str] | None = None
    url: str | None = None

    @property
    def dbsnp_url(self) -> str | None:
        """Generate dbSNP URL if rsID exists."""
        if self.rsid:
            return f"https://www.ncbi.nlm.nih.gov/snp/{self.rsid}"
        return None

    @property
    def clinvar_url(self) -> str | None:
        """Generate ClinVar URL if variant ID exists."""
        if self.variant_id:
            return f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{self.variant_id}"
        return None


class VariantSearchResponse(BaseModel):
    hits: list[VariantResult] = Field(default_factory=list)
    total: int = 0
    max_score: float | None = None


async def search_variants(query: VariantQuery, output_json: bool = False) -> str:
    """Search for genetic variants."""
    # Convert query to API parameters
    params = query.model_dump(exclude_none=True, by_alias=True)
    
    # Handle list parameters
    if "sources" in params and params["sources"]:
        params["sources"] = ",".join(params["sources"])
    
    response, error = await http_client.request_api(
        url=MYVARIANT_QUERY_URL,
        request=params,
        response_model_type=VariantSearchResponse,
        domain="variant",
    )

    if error:
        data = [{"error": f"Error {error.code}: {error.message}"}]
    else:
        data = [
            variant.model_dump(exclude_none=True, by_alias=True)
            for variant in (response.hits if response else [])
        ]

    if output_json:
        return json.dumps(data, indent=2)
    else:
        return render.to_markdown(data)


async def _variant_searcher(
    call_benefit: Annotated[
        str,
        "Define and summarize why this function is being called and the intended benefit",
    ],
    gene: Annotated[str | None, "Gene symbol"] = None,
    hgvsp: Annotated[str | None, "Protein notation"] = None,
    hgvsc: Annotated[str | None, "cDNA notation"] = None,
    rsid: Annotated[str | None, "dbSNP rsID"] = None,
    region: Annotated[str | None, "Genomic region"] = None,
    significance: Annotated[str | None, "Clinical significance"] = None,
    min_frequency: Annotated[float | None, "Minimum frequency"] = None,
    max_frequency: Annotated[float | None, "Maximum frequency"] = None,
    cadd: Annotated[float | None, "Minimum CADD score"] = None,
    polyphen: Annotated[str | None, "PolyPhen prediction"] = None,
    sift: Annotated[str | None, "SIFT prediction"] = None,
    size: Annotated[int, "Maximum number of results"] = 40,
    sources: Annotated[list[str] | str | None, "Sources to include"] = None,
) -> str:
    """
    Search for genetic variants.

    Parameters:
    - call_benefit: Define and summarize why this function is being called and the intended benefit
    - gene: Gene symbol
    - hgvsp: Protein notation
    - hgvsc: cDNA notation
    - rsid: dbSNP rsID
    - region: Genomic region
    - significance: Clinical significance
    - min_frequency: Minimum frequency
    - max_frequency: Maximum frequency
    - cadd: Minimum CADD score
    - polyphen: PolyPhen prediction
    - sift: SIFT prediction
    - size: Maximum number of results
    - sources: Sources to include

    Returns:
    Markdown formatted list of matching variants.
    """
    # Convert string sources to list
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",")]
    elif sources is None:
        sources = []

    query = VariantQuery(
        gene=gene,
        hgvsp=hgvsp,
        hgvsc=hgvsc,
        rsid=rsid,
        region=region,
        significance=significance,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        cadd=cadd,
        polyphen=polyphen,
        sift=sift,
        size=size,
        sources=sources,
    )

    return await search_variants(query, output_json=False)