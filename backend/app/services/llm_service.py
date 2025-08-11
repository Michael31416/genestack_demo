"""
LLM service supporting both OpenAI and Anthropic APIs.
"""

import os
import json
from typing import Dict, Any, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


LLM_SYSTEM_PROMPT = """You are a biomedical evidence-synthesis assistant. Assess whether the GENE is causally or mechanistically associated with the DISEASE.
Rules:
1) Use only the evidence provided. Do not invent citations or facts.
2) Weigh genetic evidence highest, then functional/omics, then literature consensus.
3) Note contradictions, biases (small N, population stratification), and whether evidence is disease-subtype-specific.
4) Prefer human data over model organisms unless human is absent.
5) Output valid JSON only matching the schema below.
6) Every claim in `key_points` must reference `source_ids` pointing to items in the input.
7) If evidence is insufficient, say so and explain next steps.

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
  "conflicts_or_gaps": [{"issue": "...", "source_ids": ["..."]}],
  "recommended_next_steps": ["..."]
}
"""


class LLMService:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider.lower()
        self.api_key = api_key
        
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def analyze_correlation(
        self, 
        evidence_json: Dict[str, Any], 
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.provider == "openai":
            return await self._call_openai(evidence_json, model or "gpt-4o-mini")
        elif self.provider == "anthropic":
            return await self._call_anthropic(evidence_json, model or "claude-3-haiku-20240307")
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
        
        user_content = (
            "Task: Evaluate correlation between {gene} and {disease}. "
            "Return JSON only. Use the provided evidence bundle.\n\n"
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
                # Retry without response_format for older models
                if e.response.status_code == 400 and "response_format" in txt.lower():
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    resp.raise_for_status()
                else:
                    raise RuntimeError(
                        f"OpenAI API error {e.response.status_code}: {txt}"
                    ) from e
            
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
                    "recommended_next_steps": [],
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
        
        user_content = (
            "Task: Evaluate correlation between gene and disease. "
            "Return ONLY valid JSON matching the schema provided in the system prompt.\n\n"
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
            resp = await client.post(
                f"{base_url}/messages",
                headers=headers,
                json=payload
            )
            
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Anthropic API error {resp.status_code}: {resp.text}"
                )
            
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
                    "recommended_next_steps": [],
                    "_raw": content
                }