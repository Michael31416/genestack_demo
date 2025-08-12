"""
LLM service supporting both OpenAI and Anthropic APIs.
"""

import os
import json
import asyncio
import time
from typing import Dict, Any, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LLMRateLimitError(Exception):
    """Raised when LLM API rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMQuotaExceededError(Exception):
    """Raised when LLM API quota is exceeded."""
    pass


class LLMAuthenticationError(Exception):
    """Raised when LLM API authentication fails."""
    pass


class LLMServiceUnavailableError(Exception):
    """Raised when LLM service is temporarily unavailable."""
    pass


# Rate limiting tracking (in-memory for simplicity)
_rate_limit_tracker = {}


def _check_rate_limit(provider: str, requests_per_minute: int = 20) -> None:
    """Check if we're within rate limits for the provider."""
    now = time.time()
    window_start = now - 60  # 1 minute window
    
    if provider not in _rate_limit_tracker:
        _rate_limit_tracker[provider] = []
    
    # Remove old requests outside the window
    _rate_limit_tracker[provider] = [
        req_time for req_time in _rate_limit_tracker[provider] 
        if req_time > window_start
    ]
    
    # Check if we're at the limit
    if len(_rate_limit_tracker[provider]) >= requests_per_minute:
        oldest_request = min(_rate_limit_tracker[provider])
        wait_time = int(61 - (now - oldest_request))  # Wait until window clears + 1 sec
        raise LLMRateLimitError(
            f"Rate limit exceeded for {provider}. Try again in {wait_time} seconds.",
            retry_after=wait_time
        )
    
    # Record this request
    _rate_limit_tracker[provider].append(now)


LLM_SYSTEM_PROMPT = """You are a biomedical evidence-synthesis assistant. Assess whether the GENE is causally or mechanistically associated with the DISEASE.
Rules:
1) Use only the evidence provided. Do not invent citations or facts.
2) Weigh genetic evidence highest, then functional/omics, then literature consensus.
3) Note contradictions, biases (small N, population stratification), and whether evidence is disease-subtype-specific.
4) Prefer human data over model organisms unless human is absent.
5) Output valid JSON only matching the schema below.
6) Every claim in `key_points` must reference `source_ids` pointing to items in the input.
7) If evidence is insufficient, say so in the `conflicts_or_gaps` section.

Schema:
{
  "verdict": "strong|moderate|weak|no_evidence|inconclusive",
  "confidence": 0.0,
  "drivers": {
    "genetic": {"present": true, "summary": "...", "source_ids": ["gwas_catalog:PMID..."]},
    "functional": {"present": true, "summary": "...", "source_ids": ["opentargets:...","literature:PMID..."]},
    "pathway_network": {"present": false, "summary": "", "source_ids": []}
  },
  "key_points": [
    {"statement": "...", "source_ids": ["gwas_catalog:PMID...", "literature:PMID..."]}
  ],
  "conflicts_or_gaps": [{"issue": "...", "source_ids": ["..."]}]
}
"""


class LLMService:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider.lower()
        self.api_key = api_key
        
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((
            httpx.TimeoutException, 
            httpx.ConnectError,
            LLMServiceUnavailableError
        ))
    )
    async def analyze_correlation(
        self, 
        evidence_json: Dict[str, Any], 
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        # Check our internal rate limits first
        try:
            _check_rate_limit(f"{self.provider}_internal", requests_per_minute=20)
        except LLMRateLimitError as e:
            # For internal rate limits, wait and then raise without retry
            await asyncio.sleep(min(e.retry_after or 60, 60))
            raise
            
        if self.provider == "openai":
            return await self._call_openai(evidence_json, model or "gpt-4o-mini")
        elif self.provider == "anthropic":
            return await self._call_anthropic(evidence_json, model or "claude-sonnet-4-20250514")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _call_openai(
        self, 
        evidence_json: Dict[str, Any], 
        model: str
    ) -> Dict[str, Any]:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Determine which data sources were included
        included_sources = []
        if evidence_json.get("opentargets"):
            included_sources.append("Open Targets Platform data")
        if evidence_json.get("literature") and len(evidence_json.get("literature", [])) > 0:
            included_sources.append("literature evidence")
        if evidence_json.get("gwas_catalog") and len(evidence_json.get("gwas_catalog", [])) > 0:
            included_sources.append("GWAS Catalog data")
        
        sources_note = ""
        if included_sources:
            sources_note = f"\n\nNOTE: The following data sources were ALREADY INCLUDED in this analysis: {', '.join(included_sources)}. Do not recommend obtaining data that has already been provided."
        
        user_content = (
            "Task: Evaluate correlation between {gene} and {disease}. "
            "Return JSON only. Use the provided evidence bundle."
            f"{sources_note}\n\n"
            f"EVIDENCE:\n{json.dumps(evidence_json, ensure_ascii=False)}"
        )
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "max_completion_tokens": 4000
        }
        
        async with httpx.AsyncClient(timeout=90) as client:
            try:
                # Try with JSON response format for newer models
                payload_with_json = dict(payload, response_format={"type": "json_object"})
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload_with_json
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                txt = e.response.text or ""
                status_code = e.response.status_code
                
                # Handle specific error types with appropriate exceptions
                if status_code == 401:
                    raise LLMAuthenticationError(f"Invalid OpenAI API key: {txt}") from e
                elif status_code == 429:
                    # Parse rate limit info from headers
                    retry_after = None
                    if 'retry-after' in e.response.headers:
                        retry_after = int(e.response.headers['retry-after'])
                    elif 'x-ratelimit-reset-requests' in e.response.headers:
                        retry_after = int(e.response.headers['x-ratelimit-reset-requests'])
                    
                    if "quota" in txt.lower() or "billing" in txt.lower():
                        raise LLMQuotaExceededError(f"OpenAI quota exceeded: {txt}") from e
                    else:
                        raise LLMRateLimitError(f"OpenAI rate limit exceeded: {txt}", retry_after) from e
                elif status_code >= 500:
                    raise LLMServiceUnavailableError(f"OpenAI service unavailable: {txt}") from e
                elif status_code == 400 and "response_format" in txt.lower():
                    # Retry without response_format for older models
                    try:
                        resp = await client.post(
                            f"{base_url}/chat/completions",
                            headers=headers,
                            json=payload
                        )
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as e2:
                        self._handle_http_error("OpenAI", e2)
                else:
                    raise RuntimeError(
                        f"OpenAI API error {status_code}: {txt}"
                    ) from e
            except httpx.TimeoutException as e:
                raise LLMServiceUnavailableError("OpenAI API timeout") from e
            except httpx.ConnectError as e:
                raise LLMServiceUnavailableError("Unable to connect to OpenAI API") from e
            
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "verdict": "inconclusive",
                    "confidence": 0.0,
                    "drivers": {},
                    "key_points": [],
                    "conflicts_or_gaps": [],
                    "_raw": content
                }
    
    async def _call_anthropic(
        self, 
        evidence_json: Dict[str, Any], 
        model: str
    ) -> Dict[str, Any]:
        base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Determine which data sources were included
        included_sources = []
        if evidence_json.get("opentargets"):
            included_sources.append("Open Targets Platform data")
        if evidence_json.get("literature") and len(evidence_json.get("literature", [])) > 0:
            included_sources.append("literature evidence")
        if evidence_json.get("gwas_catalog") and len(evidence_json.get("gwas_catalog", [])) > 0:
            included_sources.append("GWAS Catalog data")
        
        sources_note = ""
        if included_sources:
            sources_note = f"\n\nNOTE: The following data sources were ALREADY INCLUDED in this analysis: {', '.join(included_sources)}. Do not recommend obtaining data that has already been provided."
        
        user_content = (
            "Task: Evaluate correlation between gene and disease. "
            "Return ONLY valid JSON matching the schema provided in the system prompt."
            f"{sources_note}\n\n"
            f"EVIDENCE:\n{json.dumps(evidence_json, ensure_ascii=False)}"
        )
        
        payload = {
            "model": model,
            "max_tokens": 2000,
            "temperature": 0.3,
            "system": LLM_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_content}
            ]
        }
        
        async with httpx.AsyncClient(timeout=90) as client:
            try:
                resp = await client.post(
                    f"{base_url}/messages",
                    headers=headers,
                    json=payload
                )
                
                if resp.status_code != 200:
                    self._handle_anthropic_error(resp)
            except httpx.TimeoutException as e:
                raise LLMServiceUnavailableError("Anthropic API timeout") from e
            except httpx.ConnectError as e:
                raise LLMServiceUnavailableError("Unable to connect to Anthropic API") from e
            
            data = resp.json()
            content = data["content"][0]["text"]
            
            # Extract JSON from the response (Anthropic may include explanatory text)
            try:
                # Try to find JSON block in the response
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
                else:
                    return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "verdict": "inconclusive",
                    "confidence": 0.0,
                    "drivers": {},
                    "key_points": [],
                    "conflicts_or_gaps": [],
                    "_raw": content
                }
    def _handle_http_error(self, provider: str, error: httpx.HTTPStatusError) -> None:
        """Handle HTTP errors for LLM APIs."""
        txt = error.response.text or ""
        status_code = error.response.status_code
        
        if status_code == 401:
            raise LLMAuthenticationError(f"Invalid {provider} API key: {txt}") from error
        elif status_code == 429:
            retry_after = None
            if 'retry-after' in error.response.headers:
                retry_after = int(error.response.headers['retry-after'])
                
            if "quota" in txt.lower() or "billing" in txt.lower():
                raise LLMQuotaExceededError(f"{provider} quota exceeded: {txt}") from error
            else:
                raise LLMRateLimitError(f"{provider} rate limit exceeded: {txt}", retry_after) from error
        elif status_code >= 500:
            raise LLMServiceUnavailableError(f"{provider} service unavailable: {txt}") from error
        else:
            raise RuntimeError(f"{provider} API error {status_code}: {txt}") from error
    
    def _handle_anthropic_error(self, response: httpx.Response) -> None:
        """Handle Anthropic API error responses."""
        txt = response.text or ""
        status_code = response.status_code
        
        if status_code == 401:
            raise LLMAuthenticationError(f"Invalid Anthropic API key: {txt}")
        elif status_code == 429:
            retry_after = None
            if 'retry-after' in response.headers:
                retry_after = int(response.headers['retry-after'])
            
            if "quota" in txt.lower() or "billing" in txt.lower():
                raise LLMQuotaExceededError(f"Anthropic quota exceeded: {txt}")
            else:
                raise LLMRateLimitError(f"Anthropic rate limit exceeded: {txt}", retry_after)
        elif status_code >= 500:
            raise LLMServiceUnavailableError(f"Anthropic service unavailable: {txt}")
        else:
            raise RuntimeError(f"Anthropic API error {status_code}: {txt}")