import json
import logging
from ssl import TLSVersion
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from .. import StrEnum, ensure_list, http_client, render
from ..constants import CLINICAL_TRIALS_BASE_URL

logger = logging.getLogger(__name__)


class SortOrder(StrEnum):
    RELEVANCE = "RELEVANCE"
    LAST_UPDATE = "LAST_UPDATE"
    ENROLLMENT = "ENROLLMENT"
    START_DATE = "START_DATE"
    COMPLETION_DATE = "COMPLETION_DATE"
    SUBMITTED_DATE = "SUBMITTED_DATE"


class TrialPhase(StrEnum):
    EARLY_PHASE1 = "EARLY_PHASE1"
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RecruitingStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    ANY = "ANY"


class StudyType(StrEnum):
    INTERVENTIONAL = "INTERVENTIONAL"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPANDED_ACCESS = "EXPANDED_ACCESS"
    OTHER = "OTHER"


class InterventionType(StrEnum):
    DRUG = "DRUG"
    DEVICE = "DEVICE"
    BIOLOGICAL = "BIOLOGICAL"
    PROCEDURE = "PROCEDURE"
    RADIATION = "RADIATION"
    BEHAVIORAL = "BEHAVIORAL"
    GENETIC = "GENETIC"
    DIETARY_SUPPLEMENT = "DIETARY_SUPPLEMENT"
    COMBINATION_PRODUCT = "COMBINATION_PRODUCT"
    DIAGNOSTIC_TEST = "DIAGNOSTIC_TEST"
    OTHER = "OTHER"


class SponsorType(StrEnum):
    NIH = "NIH"
    OTHER_U_S_FED = "OTHER_U_S_FED"
    INDUSTRY = "INDUSTRY"
    OTHER = "OTHER"


class StudyDesign(StrEnum):
    ALLOCATION = "ALLOCATION"
    INTERVENTION_MODEL = "INTERVENTION_MODEL"
    PRIMARY_PURPOSE = "PRIMARY_PURPOSE"
    MASKING = "MASKING"
    OBSERVATIONAL_MODEL = "OBSERVATIONAL_MODEL"
    TIME_PERSPECTIVE = "TIME_PERSPECTIVE"


class AgeGroup(StrEnum):
    CHILD = "CHILD"
    ADULT = "ADULT"
    OLDER_ADULT = "OLDER_ADULT"


class PrimaryPurpose(StrEnum):
    TREATMENT = "TREATMENT"
    PREVENTION = "PREVENTION"
    DIAGNOSTIC = "DIAGNOSTIC"
    SUPPORTIVE_CARE = "SUPPORTIVE_CARE"
    SCREENING = "SCREENING"
    HEALTH_SERVICES_RESEARCH = "HEALTH_SERVICES_RESEARCH"
    BASIC_SCIENCE = "BASIC_SCIENCE"
    DEVICE_FEASIBILITY = "DEVICE_FEASIBILITY"
    OTHER = "OTHER"


class LineOfTherapy(StrEnum):
    FIRST_LINE = "FIRST_LINE"
    SECOND_LINE = "SECOND_LINE"
    THIRD_LINE = "THIRD_LINE"
    FOURTH_LINE_OR_LATER = "FOURTH_LINE_OR_LATER"
    MAINTENANCE = "MAINTENANCE"
    ADJUVANT = "ADJUVANT"
    NEOADJUVANT = "NEOADJUVANT"
    CONSOLIDATION = "CONSOLIDATION"
    SALVAGE = "SALVAGE"
    PALLIATIVE = "PALLIATIVE"
    OTHER = "OTHER"


class TrialQuery(BaseModel):
    conditions: list[str] | None = Field(default=None, description="Medical conditions")
    interventions: list[str] | None = Field(default=None, description="Interventions")
    terms: list[str] | None = Field(default=None, description="General search terms")
    nct_ids: list[str] | None = Field(default=None, description="NCT IDs")
    recruiting_status: RecruitingStatus | None = Field(default=None, description="Recruiting status")
    study_type: StudyType | None = Field(default=None, description="Study type")
    phase: TrialPhase | None = Field(default=None, description="Trial phase")
    sort: SortOrder | None = Field(default=None, description="Sort order")
    age_group: AgeGroup | None = Field(default=None, description="Age group")
    primary_purpose: PrimaryPurpose | None = Field(default=None, description="Primary purpose")
    min_date: str | None = Field(default=None, description="Minimum date (YYYY-MM-DD)")
    max_date: str | None = Field(default=None, description="Maximum date (YYYY-MM-DD)")
    date_field: str | Field(default="STUDY_START", description="Date field to filter")
    intervention_type: InterventionType | None = Field(default=None, description="Intervention type")
    sponsor_type: SponsorType | None = Field(default=None, description="Sponsor type")
    study_design: StudyDesign | None = Field(default=None, description="Study design")
    next_page_hash: str | None = Field(default=None, description="Next page hash")
    lat: float | None = Field(default=None, description="Latitude")
    long: float | None = Field(default=None, description="Longitude")
    distance: int | None = Field(default=None, description="Distance in miles")
    prior_therapies: list[str] | None = Field(default=None, description="Prior therapies")
    progression_on: list[str] | None = Field(default=None, description="Progression on therapies")
    required_mutations: list[str] | None = Field(default=None, description="Required mutations")
    excluded_mutations: list[str] | None = Field(default=None, description="Excluded mutations")
    biomarker_expression: dict[str, str] | None = Field(default=None, description="Biomarker expression")
    line_of_therapy: LineOfTherapy | None = Field(default=None, description="Line of therapy")
    allow_brain_mets: bool | None = Field(default=None, description="Allow brain metastases")
    return_fields: list[str] | None = Field(default=None, description="Return fields")
    page_size: int | Field(default=20, description="Page size")

    @field_validator("min_date", "max_date")
    @classmethod
    def validate_date(cls, v):
        if v is not None:
            from datetime import datetime
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")
        return v

    @model_validator(mode="after")
    def validate_query(self):
        if not any([self.conditions, self.interventions, self.terms, self.nct_ids]):
            raise ValueError("At least one search parameter must be provided")
        return self


class TrialResult(BaseModel):
    nct_id: str | None = None
    brief_title: str | None = None
    official_title: str | None = None
    status: str | None = None
    phase: list[str] | None = None
    study_type: str | None = None
    start_date: str | None = None
    completion_date: str | None = None
    primary_completion_date: str | None = None
    enrollment: int | None = None
    conditions: list[str] | None = None
    interventions: list[str] | None = None
    locations: list[str] | None = None
    sponsors: list[str] | None = None
    collaborators: list[str] | None = None
    study_design: dict[str, Any] | None = None
    eligibility: dict[str, Any] | None = None
    outcomes: list[dict[str, Any]] | None = None
    references: list[dict[str, Any]] | None = None
    url: str | None = None

    @property
    def title(self) -> str | None:
        """Get the best available title."""
        return self.brief_title or self.official_title

    @property
    def clinicaltrials_url(self) -> str | None:
        """Generate ClinicalTrials.gov URL."""
        if self.nct_id:
            return f"https://clinicaltrials.gov/study/{self.nct_id}"
        return None


class TrialSearchResponse(BaseModel):
    studies: list[TrialResult] = Field(default_factory=list)
    next_page_token: str | None = None
    total_studies: int = 0


async def search_trials(query: TrialQuery, output_json: bool = False) -> str:
    """Search for clinical trials."""
    # Convert query to API parameters
    params = query.model_dump(exclude_none=True, by_alias=True)
    
    # Handle list parameters
    for key, value in params.items():
        if isinstance(value, list):
            params[key] = ",".join(str(v) for v in value)
    
    # Handle dict parameters
    if "biomarker_expression" in params and params["biomarker_expression"]:
        biomarker_dict = params["biomarker_expression"]
        params["biomarker_expression"] = ",".join(f"{k}:{v}" for k, v in biomarker_dict.items())
    
    response, error = await http_client.request_api(
        url=CLINICAL_TRIALS_BASE_URL,
        request=params,
        response_model_type=TrialSearchResponse,
        domain="trial",
        tls_version=TLSVersion.TLSv1_2,
    )

    if error:
        data = [{"error": f"Error {error.code}: {error.message}"}]
    else:
        data = [
            study.model_dump(exclude_none=True, by_alias=True)
            for study in (response.studies if response else [])
        ]

    if output_json:
        return json.dumps(data, indent=2)
    else:
        return render.to_markdown(data)


async def _trial_searcher(
    call_benefit: Annotated[
        str,
        "Define and summarize why this function is being called and the intended benefit",
    ],
    conditions: Annotated[
        list[str] | str | None, "Medical conditions to search for"
    ] = None,
    interventions: Annotated[
        list[str] | str | None, "Interventions to search for"
    ] = None,
    terms: Annotated[
        list[str] | str | None, "General search terms"
    ] = None,
    nct_ids: Annotated[
        list[str] | str | None, "Specific NCT IDs to search for"
    ] = None,
    recruiting_status: Annotated[
        RecruitingStatus | str | None, "Recruiting status filter"
    ] = None,
    study_type: Annotated[
        StudyType | str | None, "Study type filter"
    ] = None,
    phase: Annotated[
        TrialPhase | str | None, "Trial phase filter"
    ] = None,
    sort: Annotated[
        SortOrder | str | None, "Sort order"
    ] = None,
    age_group: Annotated[
        AgeGroup | str | None, "Age group filter"
    ] = None,
    primary_purpose: Annotated[
        PrimaryPurpose | str | None, "Primary purpose filter"
    ] = None,
    min_date: Annotated[
        str | None, "Minimum date (YYYY-MM-DD)"
    ] = None,
    max_date: Annotated[
        str | None, "Maximum date (YYYY-MM-DD)"
    ] = None,
    date_field: Annotated[
        str, "Date field to filter"
    ] = "STUDY_START",
    intervention_type: Annotated[
        InterventionType | str | None, "Intervention type filter"
    ] = None,
    sponsor_type: Annotated[
        SponsorType | str | None, "Sponsor type filter"
    ] = None,
    study_design: Annotated[
        StudyDesign | str | None, "Study design filter"
    ] = None,
    lat: Annotated[
        float | None, "Latitude for location-based search"
    ] = None,
    long: Annotated[
        float | None, "Longitude for location-based search"
    ] = None,
    distance: Annotated[
        int | None, "Distance in miles for location-based search"
    ] = None,
    prior_therapies: Annotated[
        list[str] | str | None, "Prior therapies"
    ] = None,
    progression_on: Annotated[
        list[str] | str | None, "Progression on therapies"
    ] = None,
    required_mutations: Annotated[
        list[str] | str | None, "Required mutations"
    ] = None,
    excluded_mutations: Annotated[
        list[str] | str | None, "Excluded mutations"
    ] = None,
    biomarker_expression: Annotated[
        dict[str, str] | None, "Biomarker expression requirements"
    ] = None,
    line_of_therapy: Annotated[
        LineOfTherapy | str | None, "Line of therapy"
    ] = None,
    allow_brain_mets: Annotated[
        bool | None, "Allow brain metastases"
    ] = None,
    return_fields: Annotated[
        list[str] | str | None, "Return fields"
    ] = None,
    page_size: Annotated[
        int, "Page size"
    ] = 20,
) -> str:
    """
    Search for clinical trials.

    Parameters:
    - call_benefit: Define and summarize why this function is being called and the intended benefit
    - conditions: Medical conditions to search for
    - interventions: Interventions to search for
    - terms: General search terms
    - nct_ids: Specific NCT IDs to search for
    - recruiting_status: Recruiting status filter (OPEN, CLOSED, ANY)
    - study_type: Study type filter (INTERVENTIONAL, OBSERVATIONAL, etc.)
    - phase: Trial phase filter (PHASE1, PHASE2, PHASE3, etc.)
    - sort: Sort order (RELEVANCE, LAST_UPDATE, ENROLLMENT, etc.)
    - age_group: Age group filter (CHILD, ADULT, OLDER_ADULT)
    - primary_purpose: Primary purpose filter (TREATMENT, PREVENTION, etc.)
    - min_date: Minimum date (YYYY-MM-DD)
    - max_date: Maximum date (YYYY-MM-DD)
    - date_field: Date field to filter (STUDY_START, COMPLETION_DATE, etc.)
    - intervention_type: Intervention type filter (DRUG, DEVICE, etc.)
    - sponsor_type: Sponsor type filter (NIH, INDUSTRY, etc.)
    - study_design: Study design filter
    - lat: Latitude for location-based search
    - long: Longitude for location-based search
    - distance: Distance in miles for location-based search
    - prior_therapies: Prior therapies
    - progression_on: Progression on therapies
    - required_mutations: Required mutations
    - excluded_mutations: Excluded mutations
    - biomarker_expression: Biomarker expression requirements
    - line_of_therapy: Line of therapy
    - allow_brain_mets: Allow brain metastases
    - return_fields: Return fields
    - page_size: Page size

    Returns:
    Markdown formatted list of matching clinical trials.
    """
    # Convert string parameters to lists
    conditions = ensure_list(conditions, split_strings=True)
    interventions = ensure_list(interventions, split_strings=True)
    terms = ensure_list(terms, split_strings=True)
    nct_ids = ensure_list(nct_ids, split_strings=True)
    prior_therapies = ensure_list(prior_therapies, split_strings=True)
    progression_on = ensure_list(progression_on, split_strings=True)
    required_mutations = ensure_list(required_mutations, split_strings=True)
    excluded_mutations = ensure_list(excluded_mutations, split_strings=True)
    return_fields = ensure_list(return_fields, split_strings=True)

    # Convert string enums to enum values
    if isinstance(recruiting_status, str):
        recruiting_status = RecruitingStatus(recruiting_status.upper())
    if isinstance(study_type, str):
        study_type = StudyType(study_type.upper())
    if isinstance(phase, str):
        phase = TrialPhase(phase.upper())
    if isinstance(sort, str):
        sort = SortOrder(sort.upper())
    if isinstance(age_group, str):
        age_group = AgeGroup(age_group.upper())
    if isinstance(primary_purpose, str):
        primary_purpose = PrimaryPurpose(primary_purpose.upper())
    if isinstance(intervention_type, str):
        intervention_type = InterventionType(intervention_type.upper())
    if isinstance(sponsor_type, str):
        sponsor_type = SponsorType(sponsor_type.upper())
    if isinstance(study_design, str):
        study_design = StudyDesign(study_design.upper())
    if isinstance(line_of_therapy, str):
        line_of_therapy = LineOfTherapy(line_of_therapy.upper())

    query = TrialQuery(
        conditions=conditions,
        interventions=interventions,
        terms=terms,
        nct_ids=nct_ids,
        recruiting_status=recruiting_status,
        study_type=study_type,
        phase=phase,
        sort=sort,
        age_group=age_group,
        primary_purpose=primary_purpose,
        min_date=min_date,
        max_date=max_date,
        date_field=date_field,
        intervention_type=intervention_type,
        sponsor_type=sponsor_type,
        study_design=study_design,
        lat=lat,
        long=long,
        distance=distance,
        prior_therapies=prior_therapies,
        progression_on=progression_on,
        required_mutations=required_mutations,
        excluded_mutations=excluded_mutations,
        biomarker_expression=biomarker_expression,
        line_of_therapy=line_of_therapy,
        allow_brain_mets=allow_brain_mets,
        return_fields=return_fields,
        page_size=page_size,
    )

    return await search_trials(query, output_json=False)