"""Autocomplete functionality for biomedical entities."""

import asyncio
import json
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, Field

from .. import http_client
from ..constants import PUBTATOR3_AUTOCOMPLETE_URL

Concept = Literal["variant", "chemical", "disease", "gene"]


class EntityRequest(BaseModel):
    concept: Concept = Field(description="Type of entity to search for")
    query: str = Field(description="Search query")
    limit: int = Field(default=1, description="Maximum number of results")


class EntityResult(BaseModel):
    entity_id: str = Field(description="Unique entity identifier")
    entity_name: str = Field(description="Display name of the entity")
    concept: str = Field(description="Type of entity")
    description: str | None = Field(default=None, description="Entity description")
    synonyms: list[str] = Field(default_factory=list, description="Entity synonyms")


async def autocomplete(request: EntityRequest) -> EntityResult | None:
    """Get autocomplete suggestions for biomedical entities."""
    if not request.query.strip():
        return None

    # Build query parameters
    params = {
        "concept": request.concept,
        "query": request.query,
        "limit": request.limit,
    }

    response, error = await http_client.request_api(
        url=PUBTATOR3_AUTOCOMPLETE_URL,
        request=params,
        domain="autocomplete",
    )

    if error or not response:
        return None

    # Parse response
    if isinstance(response, list) and response:
        entity_data = response[0]
        return EntityResult(
            entity_id=entity_data.get("entity_id", ""),
            entity_name=entity_data.get("entity_name", ""),
            concept=entity_data.get("concept", request.concept),
            description=entity_data.get("description"),
            synonyms=entity_data.get("synonyms", []),
        )

    return None