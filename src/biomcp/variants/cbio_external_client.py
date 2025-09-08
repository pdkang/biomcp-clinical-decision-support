"""cBioPortal external client for variant data."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .. import http_client
from ..constants import CBIOPORTAL_BASE_URL


class CBioPortalVariantData(BaseModel):
    """cBioPortal variant data."""
    
    variant_id: str | None = None
    gene: str | None = None
    cancer_type: str | None = None
    frequency: float | None = None
    clinical_significance: str | None = None
    mutation_type: str | None = None
    protein_change: str | None = None
    

class CBioPortalClient:
    """Client for cBioPortal API."""
    
    def __init__(self, base_url: str = CBIOPORTAL_BASE_URL):
        self.base_url = base_url
    
    async def search_variants(self, gene: str, cancer_type: str | None = None) -> List[CBioPortalVariantData]:
        """Search for variants in cBioPortal."""
        try:
            params = {"gene": gene}
            if cancer_type:
                params["cancer_type"] = cancer_type
            
            response, error = await http_client.request_api(
                url=f"{self.base_url}/variants",
                request=params,
                domain="cbioportal",
            )
            
            if error or not response:
                return []
            
            if isinstance(response, list):
                return [CBioPortalVariantData(**item) for item in response]
            else:
                return [CBioPortalVariantData(**response)]
        except Exception:
            return []
    
    async def get_variant_details(self, variant_id: str) -> Optional[CBioPortalVariantData]:
        """Get detailed information about a variant."""
        try:
            response, error = await http_client.request_api(
                url=f"{self.base_url}/variants/{variant_id}",
                request={},
                domain="cbioportal",
            )
            
            if error or not response:
                return None
            
            return CBioPortalVariantData(**response)
        except Exception:
            return None
    
    async def get_cancer_types(self) -> List[str]:
        """Get list of available cancer types."""
        try:
            response, error = await http_client.request_api(
                url=f"{self.base_url}/cancer-types",
                request={},
                domain="cbioportal",
            )
            
            if error or not response:
                return []
            
            if isinstance(response, list):
                return [item.get("name", "") for item in response if isinstance(item, dict)]
            else:
                return []
        except Exception:
            return []
    
    async def get_genes(self) -> List[str]:
        """Get list of available genes."""
        try:
            response, error = await http_client.request_api(
                url=f"{self.base_url}/genes",
                request={},
                domain="cbioportal",
            )
            
            if error or not response:
                return []
            
            if isinstance(response, list):
                return [item.get("name", "") for item in response if isinstance(item, dict)]
            else:
                return []
        except Exception:
            return []