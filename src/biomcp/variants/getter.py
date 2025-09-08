"""Variant detail fetching functionality."""

import json
from typing import Annotated

from pydantic import BaseModel, Field

from .. import http_client, render
from ..constants import MYVARIANT_GET_URL


class VariantGetRequest(BaseModel):
    variant_id: str = Field(description="Variant identifier")
    include_external: bool = Field(default=True, description="Include external annotations")


class VariantDetail(BaseModel):
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
    external_annotations: dict[str, Any] | None = None
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


async def get_variant(
    variant_id: str, output_json: bool = False, include_external: bool = True
) -> str:
    """Get detailed information about a genetic variant."""
    request = VariantGetRequest(
        variant_id=variant_id, include_external=include_external
    )

    response, error = await http_client.request_api(
        url=MYVARIANT_GET_URL,
        request=request,
        response_model_type=VariantDetail,
        domain="variant",
    )

    if error:
        data = {"error": f"Error {error.code}: {error.message}"}
    else:
        data = response.model_dump(exclude_none=True) if response else {}

    if output_json:
        return json.dumps(data, indent=2)
    else:
        return render.to_markdown(data)


async def _variant_getter(
    call_benefit: Annotated[
        str,
        "Define and summarize why this function is being called and the intended benefit",
    ],
    variant_id: Annotated[str, "Variant identifier (rsID or MyVariant ID)"] = None,
    include_external: Annotated[bool, "Include external annotations"] = True,
) -> str:
    """
    Get detailed information about a genetic variant.

    Parameters:
    - call_benefit: Define and summarize why this function is being called and the intended benefit
    - variant_id: Variant identifier (rsID or MyVariant ID)
    - include_external: Include external annotations

    Returns:
    Markdown formatted variant details.
    """
    if not variant_id:
        return "Error: Variant ID is required"

    return await get_variant(variant_id, output_json=False, include_external=include_external)