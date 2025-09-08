"""Trial detail fetching functionality."""

import json
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

from .. import http_client, render
from ..constants import CLINICAL_TRIALS_STUDY_URL


class Module(Enum):
    PROTOCOL = "PROTOCOL"
    LOCATIONS = "LOCATIONS"
    OUTCOMES = "OUTCOMES"
    REFERENCES = "REFERENCES"
    ALL = "ALL"
    FULL = "FULL"


class TrialDetailRequest(BaseModel):
    nct_id: str = Field(description="NCT ID")
    module: Module = Field(default=Module.PROTOCOL, description="Module to retrieve")


class TrialDetailResponse(BaseModel):
    nct_id: str | None = None
    protocol: dict[str, Any] | None = None
    locations: list[dict[str, Any]] | None = None
    outcomes: list[dict[str, Any]] | None = None
    references: list[dict[str, Any]] | None = None
    errors: list[str] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        if "protocol" in data:
            self.protocol = data["protocol"]
        if "locations" in data:
            self.locations = data["locations"]
        if "outcomes" in data:
            self.outcomes = data["outcomes"]
        if "references" in data:
            self.references = data["references"]


async def get_trial(
    nct_id: str, module: Module = Module.PROTOCOL, output_json: bool = False
) -> str:
    """Get detailed information about a clinical trial."""
    request = TrialDetailRequest(nct_id=nct_id, module=module)

    response, error = await http_client.request_api(
        url=CLINICAL_TRIALS_STUDY_URL,
        request=request,
        response_model_type=TrialDetailResponse,
        domain="trial",
    )

    if error:
        data = {"error": f"Error {error.code}: {error.message}"}
    else:
        data = response.model_dump(exclude_none=True) if response else {}

    if output_json:
        return json.dumps(data, indent=2)
    else:
        return render.to_markdown(data)


async def _trial_getter(
    call_benefit: Annotated[
        str,
        "Define and summarize why this function is being called and the intended benefit",
    ],
    nct_id: Annotated[str, "NCT ID of the trial to retrieve"] = None,
    module: Annotated[
        Module | str, "Module to retrieve (PROTOCOL, LOCATIONS, OUTCOMES, REFERENCES, ALL, FULL)"
    ] = Module.PROTOCOL,
) -> str:
    """
    Get detailed information about a clinical trial.

    Parameters:
    - call_benefit: Define and summarize why this function is being called and the intended benefit
    - nct_id: NCT ID of the trial to retrieve
    - module: Module to retrieve (PROTOCOL, LOCATIONS, OUTCOMES, REFERENCES, ALL, FULL)

    Returns:
    Markdown formatted trial details.
    """
    if not nct_id:
        return "Error: NCT ID is required"

    # Convert string module to enum
    if isinstance(module, str):
        try:
            module = Module(module.upper())
        except ValueError:
            return f"Error: Invalid module '{module}'. Valid modules are: PROTOCOL, LOCATIONS, OUTCOMES, REFERENCES, ALL, FULL"

    return await get_trial(nct_id, module, output_json=False)