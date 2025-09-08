"""REST API server for BioMCP that wraps CLI functionality."""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import markdown2

from . import logger
from .articles.search import PubmedRequest
from .articles.unified import search_articles_unified
from .trials.search import TrialQuery, search_trials
from .variants.getter import get_variant
from .variants.search import VariantQuery, search_variants
from .thinking.sequential import _sequential_thinking
from .llm_integration import initialize_llm_integration, get_llm_orchestration, get_llm_synthesis, LLMIntegrationError

# Create FastAPI app
app = FastAPI(
    title="BioMCP REST API",
    description="REST API for BioMCP biomedical data access",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM integration
initialize_llm_integration()

# Pydantic models for request/response
class ArticleSearchRequest(BaseModel):
    genes: Optional[List[str]] = Field(default=None, description="Gene names to search for")
    diseases: Optional[List[str]] = Field(default=None, description="Disease names to search for")
    variants: Optional[List[str]] = Field(default=None, description="Genetic variants to search for")
    chemicals: Optional[List[str]] = Field(default=None, description="Chemical names to search for")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords to search for")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Results per page")
    include_preprints: bool = Field(default=True, description="Include preprint articles")
    output_json: bool = Field(default=True, description="Return JSON format")

class ArticleGetRequest(BaseModel):
    identifiers: List[str] = Field(description="Article identifiers (PMIDs or DOIs)")
    full: bool = Field(default=False, description="Fetch full article text")
    output_json: bool = Field(default=True, description="Return JSON format")

class TrialSearchRequest(BaseModel):
    conditions: Optional[List[str]] = Field(default=None, description="Medical conditions")
    interventions: Optional[List[str]] = Field(default=None, description="Treatments or interventions")
    terms: Optional[List[str]] = Field(default=None, description="General search terms")
    nct_ids: Optional[List[str]] = Field(default=None, description="Clinical trial NCT IDs")
    recruiting_status: Optional[str] = Field(default=None, description="Recruiting status")
    study_type: Optional[str] = Field(default=None, description="Study type")
    phase: Optional[str] = Field(default=None, description="Trial phase")
    sort_order: Optional[str] = Field(default=None, description="Sort order")
    age_group: Optional[str] = Field(default=None, description="Age group filter")
    primary_purpose: Optional[str] = Field(default=None, description="Primary purpose filter")
    min_date: Optional[str] = Field(default=None, description="Minimum date (YYYY-MM-DD)")
    max_date: Optional[str] = Field(default=None, description="Maximum date (YYYY-MM-DD)")
    date_field: Optional[str] = Field(default="STUDY_START", description="Date field to filter")
    intervention_type: Optional[str] = Field(default=None, description="Intervention type filter")
    sponsor_type: Optional[str] = Field(default=None, description="Sponsor type filter")
    study_design: Optional[str] = Field(default=None, description="Study design filter")
    next_page_hash: Optional[str] = Field(default=None, description="Next page hash for pagination")
    latitude: Optional[float] = Field(default=None, description="Latitude for location-based search")
    longitude: Optional[float] = Field(default=None, description="Longitude for location-based search")
    distance: Optional[int] = Field(default=None, description="Distance in miles for location-based search")
    prior_therapies: Optional[List[str]] = Field(default=None, description="Prior therapies to search for")
    progression_on: Optional[List[str]] = Field(default=None, description="Therapies the patient has progressed on")
    required_mutations: Optional[List[str]] = Field(default=None, description="Required mutations in eligibility criteria")
    excluded_mutations: Optional[List[str]] = Field(default=None, description="Excluded mutations in eligibility criteria")
    biomarker: Optional[Dict[str, str]] = Field(default=None, description="Biomarker expression requirements")
    line_of_therapy: Optional[str] = Field(default=None, description="Line of therapy filter")
    allow_brain_mets: Optional[bool] = Field(default=None, description="Whether to allow trials that accept brain metastases")
    return_field: Optional[List[str]] = Field(default=None, description="Specific fields to return")
    page_size: Optional[int] = Field(default=None, ge=1, le=1000, description="Number of results per page")
    output_json: bool = Field(default=True, description="Return JSON format")

class VariantSearchRequest(BaseModel):
    gene: Optional[str] = Field(default=None, description="Gene symbol")
    hgvsp: Optional[str] = Field(default=None, description="Protein notation")
    hgvsc: Optional[str] = Field(default=None, description="cDNA notation")
    rsid: Optional[str] = Field(default=None, description="dbSNP rsID")
    region: Optional[str] = Field(default=None, description="Genomic region")
    significance: Optional[str] = Field(default=None, description="Clinical significance")
    min_frequency: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum gnomAD exome allele frequency")
    max_frequency: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum gnomAD exome allele frequency")
    cadd: Optional[float] = Field(default=None, ge=0.0, description="Minimum CADD phred score")
    polyphen: Optional[str] = Field(default=None, description="PolyPhen-2 prediction")
    sift: Optional[str] = Field(default=None, description="SIFT prediction")
    size: int = Field(default=40, ge=1, le=100, description="Maximum number of results")
    sources: Optional[List[str]] = Field(default=None, description="Specific sources to include")
    output_json: bool = Field(default=True, description="Return JSON format")

class TrialGetRequest(BaseModel):
    nct_id: str = Field(description="Clinical trial NCT ID")
    module: Optional[str] = Field(default="PROTOCOL", description="Module to retrieve: Protocol, Locations, References, or Outcomes")
    output_json: bool = Field(default=True, description="Return JSON format")

class VariantGetRequest(BaseModel):
    variant_id: str = Field(description="Variant identifier (rsID or MyVariant ID)")
    output_json: bool = Field(default=True, description="Return JSON format")
    include_external: bool = Field(default=True, description="Include external annotations")

class AutonomousThinkRequest(BaseModel):
    query: str = Field(description="The biomedical research question or topic to analyze")
    include_articles: Optional[bool] = Field(default=None, description="Include article searches in analysis (auto-detected if not specified)")
    include_trials: Optional[bool] = Field(default=None, description="Include clinical trial searches in analysis (auto-detected if not specified)")
    include_variants: Optional[bool] = Field(default=None, description="Include genetic variant searches in analysis (auto-detected if not specified)")
    max_results_per_domain: int = Field(default=50, description="Maximum results to analyze per domain")
    samples_per_domain: int = Field(default=15, ge=1, le=50, description="Number of samples to show in synthesis per domain")
    output_json: bool = Field(default=False, description="Return JSON format (default: rich text HTML)")
    # Circuit breaker parameters
    max_thinking_steps: int = Field(default=20, ge=1, le=50, description="Maximum number of thinking steps allowed")
    max_execution_time: int = Field(default=450, ge=30, le=1800, description="Maximum execution time in seconds")
    timeout_per_step: int = Field(default=45, ge=5, le=120, description="Timeout per individual step in seconds")

def convert_markdown_to_rich_text(markdown_content: str) -> str:
    """Convert markdown to styled HTML for rich text display."""
    if not markdown_content:
        return ""
    
    logger.info(f"Converting markdown to rich text. Input length: {len(markdown_content)}")
    logger.info(f"Input preview: {markdown_content[:200]}...")
    
    # Preprocess: Handle various newline representations
    processed_content = markdown_content
    
    # Convert escaped newlines to actual newlines
    processed_content = processed_content.replace('\\n', '\n')
    
    # Handle any remaining literal \n strings that might be in the content
    processed_content = processed_content.replace('\\n\\n', '\n\n')
    processed_content = processed_content.replace('\\n', '\n')
    
    # Clean up any double newlines that might have been created
    processed_content = processed_content.replace('\n\n\n', '\n\n')
    
    # Remove any remaining literal \n strings that the LLM might have generated
    processed_content = processed_content.replace('\\n', '')
    
    html = markdown2.markdown(processed_content, extras=[
        'tables', 'fenced-code-blocks', 'code-friendly', 
        'cuddled-lists', 'break-on-newline', 'header-ids'
    ])
    
    # Post-process: Remove any remaining literal \n characters from the HTML
    # Only remove the literal string '\n', not actual newlines
    html = html.replace('\\n', '')
    
    # Also remove any remaining literal \n that might have been preserved
    # This handles cases where the LLM generates literal \n strings
    html = html.replace('\\n\\n', '')
    html = html.replace('\\n', '')
    
    logger.info(f"Conversion complete. Output length: {len(html)}")
    logger.info(f"Output preview: {html[:200]}...")
    
    return html

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "biomcp-api"}

# Article search endpoint
@app.post("/articles/search")
async def article_search(request: ArticleSearchRequest):
    """Search for biomedical research articles.
    
    Equivalent to: biomcp article search --gene BRAF --disease Melanoma
    """
    try:
        # Convert request to PubmedRequest
        pubmed_request = PubmedRequest(
            genes=request.genes or [],
            diseases=request.diseases or [],
            variants=request.variants or [],
            chemicals=request.chemicals or [],
            keywords=request.keywords or [],
        )

        # Execute search
        result = await search_articles_unified(
            pubmed_request,
            include_pubmed=True,
            include_preprints=request.include_preprints,
            output_json=request.output_json,
        )

        # Parse result
        if request.output_json:
            return json.loads(result)
        else:
            return {"result": result}

    except Exception as e:
        logger.error(f"Article search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Article search failed: {str(e)}")

# Article get endpoint
@app.post("/articles/get")
async def article_get(request: ArticleGetRequest):
    """Get detailed article information.
    
    Equivalent to: biomcp article get 21717063 --full
    """
    try:
        from .articles.fetch import fetch_articles
        from .articles.fetch import is_pmid, is_doi

        # Handle single identifier
        if len(request.identifiers) == 1:
            identifier = request.identifiers[0]
            
            # Convert to PMID if it's a number
            if is_pmid(identifier):
                pmid = int(identifier)
                result = await fetch_articles([pmid], request.full, request.output_json)
                if request.output_json:
                    return json.loads(result)
                else:
                    return {"result": result}
            elif is_doi(identifier):
                # For DOIs, we need to handle differently
                result = await fetch_articles([identifier], request.full, request.output_json)
                if request.output_json:
                    return json.loads(result)
                else:
                    return {"result": result}
            else:
                raise HTTPException(status_code=400, detail=f"Invalid identifier: {identifier}")
        else:
            # Handle multiple identifiers
            results = []
            for identifier in request.identifiers:
                try:
                    if is_pmid(identifier):
                        pmid = int(identifier)
                        article_result = await fetch_articles([pmid], False, True)
                    elif is_doi(identifier):
                        article_result = await fetch_articles([identifier], False, True)
                    else:
                        results.append({
                            "error": f"Invalid identifier: {identifier}"
                        })
                        continue
                        
                    article_data = json.loads(article_result)
                    if isinstance(article_data, list):
                        results.extend(article_data)
                    else:
                        results.append(article_data)
                except json.JSONDecodeError:
                    results.append({
                        "error": f"Failed to parse result for {identifier}"
                    })

            if request.output_json:
                return results
            else:
                from . import render
                return {"result": render.to_markdown(results)}

    except Exception as e:
        logger.error(f"Article get failed: {e}")
        raise HTTPException(status_code=500, detail=f"Article get failed: {str(e)}")

# Trial search endpoint
@app.post("/trials/search")
async def trial_search(request: TrialSearchRequest):
    """Search for clinical trials.
    
    Equivalent to: biomcp trial search --condition "Lung Cancer" --phase PHASE3
    """
    try:
        # Convert biomarker dict to expected format
        biomarker_expression = None
        if request.biomarker:
            biomarker_expression = request.biomarker

        # Create TrialQuery
        trial_query = TrialQuery(
            conditions=request.conditions,
            interventions=request.interventions,
            terms=request.terms,
            nct_ids=request.nct_ids,
            recruiting_status=request.recruiting_status,
            study_type=request.study_type,
            phase=request.phase,
            sort=request.sort_order,
            age_group=request.age_group,
            primary_purpose=request.primary_purpose,
            min_date=request.min_date,
            max_date=request.max_date,
            date_field=request.date_field,
            intervention_type=request.intervention_type,
            sponsor_type=request.sponsor_type,
            study_design=request.study_design,
            next_page_hash=request.next_page_hash,
            lat=request.latitude,
            long=request.longitude,
            distance=request.distance,
            prior_therapies=request.prior_therapies,
            progression_on=request.progression_on,
            required_mutations=request.required_mutations,
            excluded_mutations=request.excluded_mutations,
            biomarker_expression=biomarker_expression,
            line_of_therapy=request.line_of_therapy,
            allow_brain_mets=request.allow_brain_mets,
            return_fields=request.return_field,
            page_size=request.page_size,
        )

        # Execute search
        result = await search_trials(trial_query, output_json=request.output_json)

        # Parse result
        if request.output_json:
            return json.loads(result)
        else:
            return {"result": result}

    except Exception as e:
        logger.error(f"Trial search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trial search failed: {str(e)}")

# Variant search endpoint
@app.post("/variants/search")
async def variant_search(request: VariantSearchRequest):
    """Search for genetic variants.
    
    Equivalent to: biomcp variant search --gene TP53 --significance pathogenic
    """
    try:
        # Create VariantQuery
        variant_query = VariantQuery(
            gene=request.gene,
            hgvsp=request.hgvsp,
            hgvsc=request.hgvsc,
            rsid=request.rsid,
            region=request.region,
            significance=request.significance,
            min_frequency=request.min_frequency,
            max_frequency=request.max_frequency,
            cadd=request.cadd,
            polyphen=request.polyphen,
            sift=request.sift,
            size=request.size,
            sources=request.sources or [],
        )

        # Execute search
        result = await search_variants(variant_query, output_json=request.output_json)

        # Parse result
        if request.output_json:
            return json.loads(result)
        else:
            return {"result": result}

    except Exception as e:
        logger.error(f"Variant search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Variant search failed: {str(e)}")

# Trial get endpoint
@app.post("/trials/get")
async def trial_get(request: TrialGetRequest):
    """Get detailed trial information.
    
    Equivalent to: biomcp trial get NCT04280705 Protocol
    """
    try:
        from .trials.getter import get_trial, Module

        # Parse module parameter
        module = Module.PROTOCOL  # default
        if hasattr(request, 'module') and request.module:
            try:
                module = Module(request.module.upper())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid module: {request.module}")

        # Execute get
        result = await get_trial(
            request.nct_id,
            module,
            request.output_json,
        )

        # Parse result
        if request.output_json:
            return json.loads(result)
        else:
            return {"result": result}

    except Exception as e:
        logger.error(f"Trial get failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trial get failed: {str(e)}")

# Variant get endpoint
@app.post("/variants/get")
async def variant_get(request: VariantGetRequest):
    """Get detailed variant information.
    
    Equivalent to: biomcp variant get rs113488022
    """
    try:
        # Execute get
        result = await get_variant(
            request.variant_id,
            output_json=request.output_json,
            include_external=request.include_external,
        )

        # Parse result
        if request.output_json:
            return json.loads(result)
        else:
            return {"result": result}

    except Exception as e:
        logger.error(f"Variant get failed: {e}")
        raise HTTPException(status_code=500, detail=f"Variant get failed: {str(e)}")

@app.post("/thinking/autonomous")
async def autonomous_think(request: AutonomousThinkRequest):
    """LLM-Enhanced Autonomous biomedical research analysis.
    
    This endpoint uses:
    - Claude Sonnet 4 for query orchestration and search planning
    - GPT-4.1-mini for comprehensive research synthesis
    
    The system automatically:
    1. Uses LLM to parse natural language query intent
    2. Plans optimal search strategies with LLM guidance
    3. Executes searches across multiple data sources
    4. Generates comprehensive synthesis using LLM
    5. Provides professional, Claude Desktop-quality reports
    
    Circuit breaker protection:
    - Maximum thinking steps: {request.max_thinking_steps}
    - Maximum execution time: {request.max_execution_time}s
    - Timeout per step: {request.timeout_per_step}s
    
    Returns markdown-formatted analysis with title, sections, bullet points, and conclusion.
    """
    start_time = time.time()
    thinking_step_count = 0
    
    try:
        # Circuit breaker: Check if we've exceeded time limit
        def check_timeout():
            elapsed = time.time() - start_time
            if elapsed > request.max_execution_time:
                raise HTTPException(
                    status_code=408, 
                    detail=f"Circuit breaker: Execution time exceeded {request.max_execution_time}s limit (elapsed: {elapsed:.1f}s)"
                )
        
        # Circuit breaker: Check if we've exceeded step limit
        def check_step_limit():
            if thinking_step_count >= request.max_thinking_steps:
                raise HTTPException(
                    status_code=408,
                    detail=f"Circuit breaker: Maximum thinking steps ({request.max_thinking_steps}) exceeded"
                )
        
        # Step 1: LLM Query Orchestration
        check_timeout()
        check_step_limit()
        
        try:
            # Use Claude Sonnet 4 for query orchestration
            orchestration_result = await asyncio.wait_for(
                get_llm_orchestration(request.query),
                timeout=request.timeout_per_step
            )
            thinking_step_count += 1
            logger.info(f"LLM orchestration completed: {orchestration_result}")
        except LLMIntegrationError as e:
            logger.warning(f"LLM orchestration failed, falling back to rule-based approach: {e}")
            # Fallback to rule-based approach
            orchestration_result = {
                "search_domains": ["articles", "trials", "variants"],
                "query_type": "general_research",
                "extracted_parameters": {"keywords": request.query.split()}
            }
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Circuit breaker: LLM orchestration timeout exceeded {request.timeout_per_step}s"
            )
        
        # Step 2: Execute searches based on LLM orchestration
        search_results = {}
        
        # Determine which searches to perform based on LLM orchestration
        search_domains = orchestration_result.get("search_domains", [])
        extracted_params = orchestration_result.get("extracted_parameters", {})
        
        # Intelligent auto-detection with user override
        def should_include_domain(domain: str, user_preference: Optional[bool]) -> bool:
            """Determine if a domain should be included based on user preference or auto-detection."""
            if user_preference is not None:
                # User explicitly specified preference
                return user_preference
            else:
                # Auto-detect based on LLM orchestration
                return domain in search_domains
        
        # Determine which domains to search
        include_articles = should_include_domain("articles", request.include_articles)
        include_trials = should_include_domain("trials", request.include_trials)
        include_variants = should_include_domain("variants", request.include_variants)
        
        # Log the intelligent detection results
        logger.info(f"Intelligent domain detection: articles={include_articles}, trials={include_trials}, variants={include_variants}")
        logger.info(f"LLM suggested domains: {search_domains}")
        logger.info(f"User preferences: articles={request.include_articles}, trials={request.include_trials}, variants={request.include_variants}")
        
        # Step 3: Execute searches based on intelligent detection
        if include_articles:
            check_timeout()
            check_step_limit()
            
            try:
                # Use LLM-extracted parameters for article search
                article_params = extracted_params.get("article_params", {})
                genes = article_params.get("genes", [])
                diseases = article_params.get("diseases", [])
                keywords = article_params.get("keywords", request.query.split())
                
                article_search_result = await asyncio.wait_for(
                    search_articles_unified(
                        PubmedRequest(
                            genes=genes,
                            diseases=diseases,
                            keywords=keywords
                        ),
                        include_pubmed=True,
                        include_preprints=True,
                        output_json=True,
                    ),
                    timeout=request.timeout_per_step
                )
                article_data = json.loads(article_search_result)
                
                # Ensure articles are in the expected format for LLM synthesis
                if isinstance(article_data, list):
                    search_results["articles"] = {"results": article_data}
                else:
                    search_results["articles"] = article_data
                
                thinking_step_count += 1
            except asyncio.TimeoutError:
                search_results["articles"] = {"error": f"Circuit breaker: Article search timeout exceeded {request.timeout_per_step}s"}
            except Exception as e:
                search_results["articles"] = {"error": f"Article search failed: {str(e)}"}
        
        if include_trials:
            check_timeout()
            check_step_limit()
            
            try:
                # Use LLM-extracted parameters for trial search
                trial_params = extracted_params.get("trial_params", {})
                conditions = trial_params.get("conditions", [])
                phases = trial_params.get("phases", [])
                interventions = trial_params.get("interventions", [])
                
                # Debug logging
                logger.info(f"LLM extracted trial params: conditions={conditions}, phases={phases}, interventions={interventions}")
                
                # Fallback to direct query parsing if LLM extraction is insufficient
                if not conditions and "lung cancer" in request.query.lower():
                    conditions = ["Lung Cancer"]
                if not phases and "phase3" in request.query.lower():
                    phases = ["PHASE3"]
                
                logger.info(f"Final trial search params: conditions={conditions}, phases={phases}")
                
                trial_query = TrialQuery(
                    conditions=conditions,
                    phase=phases[0] if phases else None,
                    interventions=interventions,
                    page_size=request.max_results_per_domain * 2
                )
                
                trial_search_result = await asyncio.wait_for(
                    search_trials(trial_query, output_json=True),
                    timeout=request.timeout_per_step
                )
                trial_data = json.loads(trial_search_result)
                
                # Ensure trials are in the expected format for LLM synthesis
                if isinstance(trial_data, list):
                    search_results["trials"] = {"results": trial_data}
                else:
                    search_results["trials"] = trial_data
                
                thinking_step_count += 1
            except asyncio.TimeoutError:
                search_results["trials"] = {"error": f"Circuit breaker: Trial search timeout exceeded {request.timeout_per_step}s"}
            except Exception as e:
                search_results["trials"] = {"error": f"Trial search failed: {str(e)}"}
        
        if include_variants:
            check_timeout()
            check_step_limit()
            
            try:
                # Use LLM-extracted parameters for variant search
                variant_params = extracted_params.get("variant_params", {})
                genes = variant_params.get("genes", [])
                significance = variant_params.get("significance", [])
                
                # Use first gene if available, otherwise fall back to query terms
                gene = genes[0] if genes else None
                sig = significance[0] if significance else None
                
                variant_query = VariantQuery(
                    gene=gene,
                    significance=sig,
                    size=request.max_results_per_domain * 2
                )
                
                variant_search_result = await asyncio.wait_for(
                    search_variants(variant_query, output_json=True),
                    timeout=request.timeout_per_step
                )
                variant_data = json.loads(variant_search_result)
                
                # Ensure variants are in the expected format for LLM synthesis
                if isinstance(variant_data, list):
                    search_results["variants"] = {"results": variant_data}
                else:
                    search_results["variants"] = variant_data
                
                thinking_step_count += 1
            except asyncio.TimeoutError:
                search_results["variants"] = {"error": f"Circuit breaker: Variant search timeout exceeded {request.timeout_per_step}s"}
            except Exception as e:
                search_results["variants"] = {"error": f"Variant search failed: {str(e)}"}
        
        # Step 4: LLM Synthesis Generation
        check_timeout()
        check_step_limit()
        
        # Step 4: LLM Synthesis Generation
        try:
            # Determine synthesis type based on LLM orchestration
            query_type = orchestration_result.get("query_type", "general_research")
            
            # Use GPT-4.1-mini for synthesis
            synthesis = await asyncio.wait_for(
                get_llm_synthesis(request.query, search_results, query_type, request.samples_per_domain),
                timeout=request.timeout_per_step
            )
            thinking_step_count += 1
            logger.info(f"LLM synthesis completed for query type: {query_type}")
            
        except (LLMIntegrationError, asyncio.TimeoutError) as e:
            logger.error(f"LLM synthesis failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"LLM synthesis failed: {str(e)}. Please check API keys and try again."
            )
        
        # Convert synthesis to rich text HTML
        rich_text_synthesis = convert_markdown_to_rich_text(synthesis)
        
        # Return the LLM-generated synthesis in rich text format
        if request.output_json:
            return {
                "synthesis": rich_text_synthesis,
                "format": "rich_text",
                "query": request.query,
                "search_domains": search_domains,
                "extracted_params": extracted_params
            }
        else:
            return {
                "result": rich_text_synthesis,
                "format": "rich_text"
            }
            
    except Exception as e:
        logger.error(f"Autonomous thinking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Autonomous thinking failed: {str(e)}")

# Convenience endpoints for common CLI patterns
@app.get("/articles/search")
async def article_search_get(
    genes: Optional[str] = Query(None, description="Gene names (comma-separated)"),
    diseases: Optional[str] = Query(None, description="Disease names (comma-separated)"),
    variants: Optional[str] = Query(None, description="Genetic variants (comma-separated)"),
    chemicals: Optional[str] = Query(None, description="Chemical names (comma-separated)"),
    keywords: Optional[str] = Query(None, description="Keywords (comma-separated)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    include_preprints: bool = Query(True, description="Include preprint articles"),
    output_json: bool = Query(True, description="Return JSON format")
):
    """GET endpoint for article search with query parameters."""
    # Convert comma-separated strings to lists
    def parse_list_param(param: Optional[str]) -> Optional[List[str]]:
        if param is None:
            return None
        return [item.strip() for item in param.split(",") if item.strip()]

    request = ArticleSearchRequest(
        genes=parse_list_param(genes),
        diseases=parse_list_param(diseases),
        variants=parse_list_param(variants),
        chemicals=parse_list_param(chemicals),
        keywords=parse_list_param(keywords),
        page=page,
        page_size=page_size,
        include_preprints=include_preprints,
        output_json=output_json,
    )
    
    return await article_search(request)

@app.get("/articles/get/{identifier}")
async def article_get_single(
    identifier: str,
    full: bool = Query(False, description="Fetch full article text"),
    output_json: bool = Query(True, description="Return JSON format")
):
    """GET endpoint for single article retrieval."""
    request = ArticleGetRequest(
        identifiers=[identifier],
        full=full,
        output_json=output_json,
    )
    
    return await article_get(request)

@app.get("/trials/search")
async def trial_search_get(
    conditions: Optional[str] = Query(None, description="Medical conditions (comma-separated)"),
    interventions: Optional[str] = Query(None, description="Interventions (comma-separated)"),
    phase: Optional[str] = Query(None, description="Trial phase"),
    recruiting_status: Optional[str] = Query(None, description="Recruiting status"),
    output_json: bool = Query(True, description="Return JSON format")
):
    """GET endpoint for trial search with query parameters."""
    def parse_list_param(param: Optional[str]) -> Optional[List[str]]:
        if param is None:
            return None
        return [item.strip() for item in param.split(",") if item.strip()]

    request = TrialSearchRequest(
        conditions=parse_list_param(conditions),
        interventions=parse_list_param(interventions),
        phase=phase,
        recruiting_status=recruiting_status,
        output_json=output_json,
    )
    
    return await trial_search(request)

@app.get("/trials/get/{nct_id}")
async def trial_get_single(
    nct_id: str,
    module: Optional[str] = Query("PROTOCOL", description="Module to retrieve: Protocol, Locations, References, or Outcomes"),
    output_json: bool = Query(True, description="Return JSON format")
):
    """GET endpoint for single trial retrieval."""
    request = TrialGetRequest(
        nct_id=nct_id,
        module=module,
        output_json=output_json,
    )
    
    return await trial_get(request)

@app.get("/variants/get/{variant_id}")
async def variant_get_single(
    variant_id: str,
    output_json: bool = Query(True, description="Return JSON format"),
    include_external: bool = Query(True, description="Include external annotations")
):
    """GET endpoint for single variant retrieval."""
    request = VariantGetRequest(
        variant_id=variant_id,
        output_json=output_json,
        include_external=include_external,
    )
    
    return await variant_get(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)