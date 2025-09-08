"""External variant data sources."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .. import http_client
from ..constants import (
    CBIOPORTAL_BASE_URL,
    CLINVAR_BASE_URL,
    COSMIC_BASE_URL,
    ENSEMBL_VARIATION_URL,
)


class ExternalVariantData(BaseModel):
    """External variant data from various sources."""
    
    source: str = Field(description="Data source")
    variant_id: str = Field(description="Variant identifier")
    data: Dict[str, Any] = Field(description="Variant data")
    url: Optional[str] = Field(default=None, description="Source URL")


class ClinVarData(BaseModel):
    """ClinVar variant data."""
    
    variant_id: str | None = None
    clinical_significance: str | None = None
    review_status: str | None = None
    last_evaluated: str | None = None
    submitter: str | None = None
    condition: str | None = None
    inheritance: str | None = None
    

class COSMICData(BaseModel):
    """COSMIC variant data."""
    
    variant_id: str | None = None
    gene: str | None = None
    mutation_type: str | None = None
    mutation_description: str | None = None
    primary_site: str | None = None
    histology: str | None = None
    frequency: float | None = None


class EnsemblData(BaseModel):
    """Ensembl variant data."""
    
    variant_id: str | None = None
    chromosome: str | None = None
    position: int | None = None
    ref: str | None = None
    alt: str | None = None
    consequence: str | None = None
    impact: str | None = None
    

class CBioPortalData(BaseModel):
    """cBioPortal variant data."""
    
    variant_id: str | None = None
    gene: str | None = None
    cancer_type: str | None = None
    frequency: float | None = None
    clinical_significance: str | None = None
    

async def fetch_clinvar_data(variant_id: str) -> Optional[ClinVarData]:
    """Fetch ClinVar data for a variant."""
    try:
        response, error = await http_client.request_api(
            url=f"{CLINVAR_BASE_URL}/{variant_id}",
            request={},
            domain="external",
        )
        
        if error or not response:
            return None
        
        return ClinVarData(**response)
    except Exception:
        return None


async def fetch_cosmic_data(variant_id: str) -> Optional[COSMICData]:
    """Fetch COSMIC data for a variant."""
    try:
        response, error = await http_client.request_api(
            url=f"{COSMIC_BASE_URL}{variant_id}",
            request={},
            domain="external",
        )
        
        if error or not response:
            return None
        
        return COSMICData(**response)
    except Exception:
        return None


async def fetch_ensembl_data(variant_id: str) -> Optional[EnsemblData]:
    """Fetch Ensembl data for a variant."""
    try:
        response, error = await http_client.request_api(
            url=f"{ENSEMBL_VARIATION_URL}/{variant_id}",
            request={},
            domain="external",
        )
        
        if error or not response:
            return None
        
        return EnsemblData(**response)
    except Exception:
        return None


async def fetch_cbioportal_data(variant_id: str) -> Optional[CBioPortalData]:
    """Fetch cBioPortal data for a variant."""
    try:
        response, error = await http_client.request_api(
            url=f"{CBIOPORTAL_BASE_URL}/variants/{variant_id}",
            request={},
            domain="external",
        )
        
        if error or not response:
            return None
        
        return CBioPortalData(**response)
    except Exception:
        return None


async def fetch_all_external_data(variant_id: str) -> List[ExternalVariantData]:
    """Fetch data from all external sources."""
    tasks = [
        fetch_clinvar_data(variant_id),
        fetch_cosmic_data(variant_id),
        fetch_ensembl_data(variant_id),
        fetch_cbioportal_data(variant_id),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    external_data = []
    sources = ["clinvar", "cosmic", "ensembl", "cbioportal"]
    
    for i, result in enumerate(results):
        if isinstance(result, Exception) or result is None:
            continue
        
        external_data.append(ExternalVariantData(
            source=sources[i],
            variant_id=variant_id,
            data=result.model_dump(exclude_none=True),
            url=f"https://example.com/{sources[i]}/{variant_id}"
        ))
    
    return external_data