"""LLM integration module for BioMCP API server.

This module provides integration with:
- Claude Sonnet 4 (latest) for orchestration (query parsing, search planning)
- GPT-4.1-mini for synthesis (generating comprehensive research reports)
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union
import httpx
from . import logger

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not available, continue without it
    pass

class LLMIntegrationError(Exception):
    """Custom exception for LLM integration errors."""
    pass

class ClaudeOrchestrator:
    """Claude Sonnet 4 (latest) integration for query orchestration and search planning."""
    
    def __init__(self, api_key: Optional[str] = None):
        # Try to get API key from .env file first, then system environment
        if api_key:
            self.api_key = api_key
        else:
            # Load from .env file directly
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)  # Override system environment variables
            except ImportError:
                pass
            
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise LLMIntegrationError("ANTHROPIC_API_KEY environment variable is required for Claude orchestration")
        
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-sonnet-4-20250514"
    
    async def parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Parse natural language query to determine search intent and parameters."""
        
        system_prompt = """You are an expert biomedical research assistant. Your task is to analyze a natural language query and extract structured search parameters.

            Analyze the query and return a JSON object with the following structure:
            {
                "search_domains": ["articles", "trials", "variants"],
                "extracted_parameters": {
                    "genes": ["list of gene names"],
                    "diseases": ["list of disease names"],
                    "variants": ["list of variant names"],
                    "trial_phases": ["PHASE1", "PHASE2", "PHASE3"],
                    "trial_conditions": ["list of medical conditions"],
                    "trial_interventions": ["list of interventions"],
                    "variant_significance": ["pathogenic", "likely_pathogenic", "benign"],
                    "keywords": ["list of search keywords"]
                },
                "query_type": "general_research|trial_focused|variant_focused",
                "confidence": 0.0-1.0
            }

            INTELLIGENT DOMAIN DETECTION RULES:
            - Include "articles" for: research queries, literature reviews, general biomedical topics
            - Include "trials" for: clinical trial queries, phase mentions, recruitment status, treatment studies
            - Include "variants" for: gene-specific queries, mutation analysis, genetic variants, pathogenic significance
            - For comprehensive research: include all relevant domains
            - For focused queries: include only the most relevant domain(s)

            Examples:
            - "BRAF mutations in melanoma" → {"search_domains": ["articles", "trials", "variants"], "query_type": "general_research"}
            - "search in PHASE3 with condition on Lung Cancer" → {"search_domains": ["trials"], "query_type": "trial_focused"}
            - "search on gene TP53 with pathogenic significance" → {"search_domains": ["variants", "articles"], "query_type": "variant_focused"}
            - "EGFR mutations in lung cancer treatment" → {"search_domains": ["articles", "trials", "variants"], "query_type": "general_research"}

            Be precise and only include parameters that are explicitly mentioned in the query."""

        user_prompt = f"Analyze this biomedical research query: '{query}'"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 8000,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}]
                    }
                )
                
                if response.status_code != 200:
                    raise LLMIntegrationError(f"Claude API error: {response.status_code} - {response.text}")
                
                result = response.json()
                content = result["content"][0]["text"]
                
                # Log response length for monitoring
                logger.info(f"Claude response length: {len(content)} characters")
                
                # Extract JSON from Claude's response
                try:
                    # Find JSON object in the response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        parsed_result = json.loads(json_str)
                        return parsed_result
                    else:
                        raise LLMIntegrationError("No JSON found in Claude response")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Claude response as JSON: {content}")
                    raise LLMIntegrationError(f"Invalid JSON response from Claude: {e}")
                    
        except httpx.TimeoutException:
            raise LLMIntegrationError("Claude API request timed out")
        except Exception as e:
            raise LLMIntegrationError(f"Claude API request failed: {str(e)}")

class GPTSynthesizer:
    """GPT-4.1-mini integration for generating comprehensive research synthesis."""
    
    def __init__(self, api_key: Optional[str] = None):
        # Try to get API key from .env file first, then system environment
        if api_key:
            self.api_key = api_key
        else:
            # Load from .env file directly
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)  # Override system environment variables
            except ImportError:
                pass
            
            self.api_key = os.getenv("BIOMCP_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise LLMIntegrationError("OPENAI_API_KEY or BIOMCP_OPENAI_API_KEY environment variable is required for GPT synthesis")
        
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4.1-mini"
    
    async def generate_synthesis(self, query: str, search_results: Dict[str, Any], synthesis_type: str = "general", samples_per_domain: int = 5) -> str:
        """Generate comprehensive research synthesis using GPT-4.1-mini."""
        
        # Prepare context from search results
        context_data = self._prepare_synthesis_context(search_results, samples_per_domain)
        
        if synthesis_type == "trial_focused":
            system_prompt = self._get_trial_synthesis_prompt()
        elif synthesis_type == "variant_focused":
            system_prompt = self._get_variant_synthesis_prompt()
        else:
            system_prompt = self._get_general_synthesis_prompt()
        
        user_prompt = f"""Research Query: {query}

Search Results Summary:
{json.dumps(context_data, indent=2)}

IMPORTANT: You must ONLY report on the actual search results provided above. Do NOT generate fictional or generic content. If the search results are empty or contain errors, clearly state this in your report.

Generate a comprehensive, professional research synthesis that:
1. Provides an executive summary with key findings from the ACTUAL search results
2. Analyzes the data quantitatively where possible
3. Identifies trends and patterns from the ACTUAL data
4. Highlights clinical implications based on the ACTUAL findings
5. Suggests future research directions based on the ACTUAL gaps identified
6. Uses proper markdown formatting with headers, bullet points, and structured sections

CRITICAL: Only include information that is directly supported by the search results provided. If the search results don't contain relevant data, clearly state this limitation."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 8000,
                        "temperature": 0.3
                    }
                )
                
                if response.status_code != 200:
                    raise LLMIntegrationError(f"OpenAI API error: {response.status_code} - {response.text}")
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Log response length for monitoring
                logger.info(f"GPT synthesis response length: {len(content)} characters")
                
                return content
                
        except httpx.TimeoutException:
            raise LLMIntegrationError("OpenAI API request timed out")
        except Exception as e:
            raise LLMIntegrationError(f"OpenAI API request failed: {str(e)}")
    
    def _prepare_synthesis_context(self, search_results: Dict[str, Any], samples_per_domain: int = 5) -> Dict[str, Any]:
        """Prepare search results for synthesis context."""
        context = {}
        
        for domain, results in search_results.items():
            if isinstance(results, dict) and 'results' in results:
                domain_results = results['results']
                if isinstance(domain_results, list):
                    # Extract key information from first few results
                    context[domain] = {
                        "count": len(domain_results),
                        "sample_data": domain_results[:samples_per_domain] if len(domain_results) > 0 else []
                    }
                else:
                    context[domain] = {"count": 0, "sample_data": []}
            elif isinstance(results, list):
                context[domain] = {
                    "count": len(results),
                    "sample_data": results[:samples_per_domain] if len(results) > 0 else []
                }
            else:
                context[domain] = {"count": 0, "sample_data": []}
        
        return context
    
    def _get_general_synthesis_prompt(self) -> str:
        return """You are an expert biomedical research analyst. Your task is to synthesize research findings into a comprehensive, professional report.

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:
1. You must ONLY report on the actual search results provided
2. Do NOT generate fictional or generic content
3. Analyze the EXACT data provided in the search results
4. If the search results contain specific articles, trials, or variants, list them with their actual details
5. If no relevant data is found, clearly state this
6. NEVER invent or hallucinate information
7. ONLY use the exact data provided in the search results

Structure your response with:
1. **Executive Summary** - Key findings and implications
2. **Research Context** - Background and significance
3. **Key Findings** - Quantitative and qualitative analysis
4. **Clinical Implications** - Practical applications
5. **Future Directions** - Research gaps and opportunities
6. **Conclusion** - Summary of impact
7. **References** - Detailed citations with exact information

For each article mentioned, include:
- Author names (exact from search results)
- Publication year (exact from search results)
- Article title (exact from search results)
- DOI link (if available in search results)
- Journal name (if available in search results)

For each trial mentioned, include:
- NCT ID (exact from search results)
- Study Title (exact from search results)
- Phase (exact from search results)
- Status (exact from search results)
- ClinicalTrials.gov link (if available)

For each variant mentioned, include:
- Gene name (exact from search results)
- Variant ID (exact from search results)
- Clinical significance (exact from search results)
- CADD score (if available in search results)

Use markdown formatting with headers, bullet points, and structured sections. Be specific, quantitative, and actionable. Always include a detailed References section with exact citations."""

    def _get_trial_synthesis_prompt(self) -> str:
        return """You are an expert clinical trial analyst. Focus on synthesizing clinical trial data into a structured report.

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:
1. You must ONLY report on the actual trials provided in the search results
2. Do NOT generate fictional trials or generic content
3. Analyze the EXACT data provided in the search results
4. If the search results contain specific trials, list them with their actual details
5. If no relevant trials are found, clearly state this
6. NEVER invent or hallucinate trial information
7. ONLY use the exact trial data provided in the search results

Structure your response with:
1. **Currently Recruiting Trials** - List the actual trials found with their specific details
2. **Recently Completed Trials** - List any completed trials from the results
3. **Key Treatment Categories** - Analyze the actual therapeutic approaches found
4. **Clinical Impact** - Implications based on the actual trials found
5. **Research Gaps** - Areas needing more trials based on the actual results

For each trial, include:
- NCT ID (exact from search results)
- Study Title (exact from search results)
- Phase (exact from search results)
- Status (exact from search results)
- Enrollment numbers (exact from search results)
- Key interventions being studied (exact from search results)

IMPORTANT: Only include trials that are actually present in the search results provided. Do not make up or generate any trials that are not in the data. If you cannot find specific details in the search results, state "Information not available in search results" rather than inventing data."""

    def _get_variant_synthesis_prompt(self) -> str:
        return """You are an expert genetic variant analyst. Focus on synthesizing genetic variant data into a comprehensive report.

Structure your response with:
1. **Cancer Genomics Context** - Gene's role in cancer
2. **Key Pathogenic Variants** - Categorized by type (nonsense, missense, splice)
3. **Clinical Significance** - Disease associations
4. **Functional Predictions** - CADD scores, PolyPhen, SIFT
5. **Population Frequency** - gnomAD data
6. **Key Observations** - Important insights

Provide specific variant details with proper genetic notation and clinical context."""

# Global instances
claude_orchestrator: Optional[ClaudeOrchestrator] = None
gpt_synthesizer: Optional[GPTSynthesizer] = None

def initialize_llm_integration():
    """Initialize LLM integration with Claude and GPT."""
    global claude_orchestrator, gpt_synthesizer
    
    # Always load from .env file first, overriding system environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)  # Override system environment variables
        logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning("python-dotenv not available, using system environment variables")
    except Exception as e:
        logger.warning(f"Failed to load .env file: {e}")
    
    try:
        claude_orchestrator = ClaudeOrchestrator()
        logger.info("Claude Sonnet 4 orchestrator initialized successfully")
    except LLMIntegrationError as e:
        logger.warning(f"Claude orchestrator initialization failed: {e}")
        claude_orchestrator = None
    
    try:
        gpt_synthesizer = GPTSynthesizer()
        logger.info("GPT-4.1-mini synthesizer initialized successfully")
    except LLMIntegrationError as e:
        logger.warning(f"GPT synthesizer initialization failed: {e}")
        gpt_synthesizer = None

async def get_llm_orchestration(query: str) -> Dict[str, Any]:
    """Get LLM orchestration for a query."""
    if not claude_orchestrator:
        raise LLMIntegrationError("Claude orchestrator not initialized")
    
    return await claude_orchestrator.parse_query_intent(query)

async def get_llm_synthesis(query: str, search_results: Dict[str, Any], synthesis_type: str = "general", samples_per_domain: int = 5) -> str:
    """Get LLM synthesis for search results."""
    if not gpt_synthesizer:
        raise LLMIntegrationError("GPT synthesizer not initialized")
    return await gpt_synthesizer.generate_synthesis(query, search_results, synthesis_type, samples_per_domain)