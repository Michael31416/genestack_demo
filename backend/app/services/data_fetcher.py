"""
Data fetching services adapted from the CLI tool.
Handles fetching from Ensembl, OLS, Open Targets, Europe PMC, and GWAS Catalog.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import (
    HTTP_TIMEOUT_SHORT, HTTP_TIMEOUT_MEDIUM,
    DEFAULT_ENSEMBL_ROWS, DEFAULT_OPENTARGETS_SIZE, DEFAULT_LITERATURE_PAGE_SIZE,
    MAX_GWAS_RECORDS, LITERATURE_FUTURE_YEAR,
    RETRY_MIN_WAIT_SECONDS, RETRY_MAX_WAIT_SECONDS
)


OT_GQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
ENSEMBL_LOOKUP = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}?content-type=application/json"
OLS_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api"

HEADERS_JSON = {"Accept": "application/json"}
HEADERS_GQL = {"Content-Type": "application/json", "Accept": "application/json"}


@retry(wait=wait_exponential(multiplier=0.5, min=RETRY_MIN_WAIT_SECONDS, max=RETRY_MAX_WAIT_SECONDS), stop=stop_after_attempt(3))
async def fetch_json(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    method: str = "GET",
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    headers = headers or HEADERS_JSON
    if method.upper() == "GET":
        resp = await client.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT_SHORT)
    else:
        resp = await client.post(
            url, params=params, headers=headers, json=json_body, timeout=HTTP_TIMEOUT_MEDIUM
        )
    resp.raise_for_status()
    return resp.json()


async def resolve_gene_symbol(
    client: httpx.AsyncClient, symbol: str
) -> Tuple[str, List[str]]:
    """Return (ensembl_id, synonyms)."""
    url = ENSEMBL_LOOKUP.format(symbol=symbol)
    try:
        data = await fetch_json(client, url)
    except Exception:
        raise RuntimeError(f"Could not resolve gene symbol '{symbol}' via Ensembl.")
    ensg = data.get("id")
    synonyms = list({data.get("display_name", symbol), symbol})
    return ensg, [s for s in synonyms if s]


async def resolve_disease_label(
    client: httpx.AsyncClient, label: str
) -> Tuple[str, Optional[str], List[str]]:
    """Return (efo_id, mondo_id, synonyms) using OLS4 search."""
    params = {
        "q": label,
        "ontology": "efo,mondo",
        "type": "class",
        "rows": DEFAULT_ENSEMBL_ROWS,
        "exact": "false",
    }
    data = await fetch_json(client, OLS_SEARCH, params=params)
    docs: List[Dict[str, Any]] = []
    if "response" in data and "docs" in data["response"]:
        docs = data["response"]["docs"]
    elif "_embedded" in data and "terms" in data["_embedded"]:
        docs = data["_embedded"]["terms"]
    best_efo = None
    best_mondo = None
    for d in docs:
        onto = d.get("ontology_name") or d.get("ontology_prefix") or d.get("ontology")
        short = d.get("short_form") or d.get("obo_id", "").split("/")[-1]
        lbl = d.get("label") or d.get("name") or ""
        syns = d.get("synonym", []) or d.get("synonyms", [])
        entry = {"id": short, "label": lbl, "synonyms": syns}
        if onto and onto.lower().startswith("efo") and not best_efo:
            best_efo = entry
        if onto and onto.lower().startswith("mondo") and not best_mondo:
            best_mondo = entry
    efo_id = (
        f"EFO_{best_efo['id']}"
        if best_efo and not best_efo["id"].startswith("EFO_")
        else (best_efo["id"] if best_efo else None)
    )
    mondo_id = (
        f"MONDO:{best_mondo['id']}"
        if best_mondo and not best_mondo["id"].startswith("MONDO")
        else (best_mondo["id"] if best_mondo else None)
    )
    if not efo_id and not mondo_id:
        raise RuntimeError(f"Could not resolve disease '{label}' via OLS.")
    syns = list(
        {
            *(best_efo.get("synonyms", []) if best_efo else []),
            *(best_mondo.get("synonyms", []) if best_mondo else []),
            label,
        }
    )
    return efo_id or "", mondo_id, syns


OT_ASSOCIATED_TARGETS_QUERY = """
query DiseaseTargets($efoId: String!, $index: Int!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: $index, size: $size}) {
      count
      rows {
        score
        target { id approvedSymbol }
        datatypeScores { id score }
      }
    }
  }
}
"""


async def get_opentargets_association(
    client: httpx.AsyncClient, ensg: str, efo_id: str
) -> Optional[Dict[str, Any]]:
    index = 0
    size = DEFAULT_OPENTARGETS_SIZE
    total = None
    while True:
        variables = {"efoId": efo_id, "index": index, "size": size}
        body = {"query": OT_ASSOCIATED_TARGETS_QUERY, "variables": variables}
        try:
            data = await fetch_json(
                client, OT_GQL_URL, headers=HEADERS_GQL, method="POST", json_body=body
            )
        except Exception:
            break
        disease = (data.get("data") or {}).get("disease") or {}
        assoc = disease.get("associatedTargets") or {}
        rows = assoc.get("rows") or []
        for r in rows:
            tgt = r.get("target") or {}
            if tgt.get("id") == ensg:
                return {
                    "overall_association_score": r.get("score"),
                    "datatype_scores": r.get("datatypeScores"),
                    "disease": {"id": disease.get("id"), "name": disease.get("name")},
                    "target": {"id": ensg},
                }
        total = assoc.get("count") or 0
        index += 1
        if index * size >= total:
            break
    return None


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    protected = {
        "e.g.": "eg",
        "i.e.": "ie",
        "et al.": "etal",
        "Fig.": "Fig",
        "Dr.": "Dr",
    }
    for k, v in protected.items():
        text = text.replace(k, v)
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9])", text)
    for i in range(len(parts)):
        for k, v in protected.items():
            parts[i] = parts[i].replace(v, k)
    return [p.strip() for p in parts if p.strip()]


async def get_literature_evidence(
    client: httpx.AsyncClient,
    gene_terms: List[str],
    disease_terms: List[str],
    since_year: int,
    max_records: int,
) -> List[Dict[str, Any]]:
    gene_q = " OR ".join([f'"{t}"' if " " in t else t for t in gene_terms[:5]])
    dis_q = " OR ".join([f'"{t}"' if " " in t else t for t in disease_terms[:8]])
    query = f"({gene_q}) AND ({dis_q}) AND (PUB_YEAR:[{since_year} TO {LITERATURE_FUTURE_YEAR}])"
    params = {
        "query": query,
        "resultType": "core",
        "pageSize": min(max_records, DEFAULT_LITERATURE_PAGE_SIZE),
        "format": "json",
    }
    data = await fetch_json(client, EUROPE_PMC_SEARCH, params=params)
    hits = (data.get("resultList") or {}).get("result", []) or []
    out = []
    for h in hits[:max_records]:
        pmid = h.get("pmid") or h.get("id")
        title = h.get("title", "")
        year = int(h.get("pubYear")) if h.get("pubYear") else None
        src = h.get("source", "")
        author = h.get("authorString", "")
        abstract = h.get("abstractText") or ""
        sents = split_sentences(abstract)
        g_pat = re.compile("|".join([re.escape(t) for t in gene_terms]), re.I)
        d_pat = re.compile("|".join([re.escape(t) for t in disease_terms]), re.I)
        evidence_sents = [s for s in sents if g_pat.search(s) and d_pat.search(s)]
        if not evidence_sents and title and g_pat.search(title) and d_pat.search(title):
            evidence_sents = [title]
        if not evidence_sents:
            continue
        out.append(
            {
                "pmid": pmid,
                "title": title,
                "year": year,
                "source": src,
                "author": author,
                "sentences": evidence_sents[:3],
                "uri": f"https://europepmc.org/abstract/MED/{pmid}" if pmid else None,
            }
        )
    return out


async def get_gwas_associations(
    client: httpx.AsyncClient,
    efo_label_or_id: str,
    gene_symbol: str,
    max_records: int = 20,
) -> List[Dict[str, Any]]:
    params = {"efoTrait": efo_label_or_id, "size": min(max_records, MAX_GWAS_RECORDS)}
    url = f"{GWAS_API_BASE}/associations/search"
    try:
        data = await fetch_json(client, url, params=params)
    except Exception:
        return []
    embedded = data.get("_embedded", {})
    assocs = embedded.get("associations", []) if embedded else []
    out = []
    for a in assocs:
        loci = a.get("loci", [])
        keep = False
        mapped_genes: List[str] = []
        for locus in loci:
            rs = locus.get("strongestRiskAlleles", [])
            _genes: List[str] = []
            for sra in rs:
                _genes.extend(
                    [
                        g.get("geneName")
                        for g in (sra.get("ensemblGenes") or [])
                        if g.get("geneName")
                    ]
                )
            mapped = locus.get("authorReportedGenes", [])
            if isinstance(mapped, list):
                mapped_genes.extend(mapped)
            elif isinstance(mapped, str):
                mapped_genes.extend([g.strip() for g in mapped.split(",")])
            mapped_genes.extend([g for g in _genes if g])
        if gene_symbol in set(mapped_genes):
            keep = True
        if keep:
            out.append(
                {
                    "association_id": a.get("associationId"),
                    "pvalue": a.get("pvalueMantissa"),
                    "pvalueExponent": a.get("pvalueExponent"),
                    "or_or_beta": a.get("orPerCopyNum") or a.get("betaNum"),
                    "ci": a.get("ci"),
                    "pmid": a.get("pubmedId"),
                    "trait": a.get("trait"),
                    "study_accession": a.get("studyAccession"),
                    "uri": a.get("_links", {}).get("self", {}).get("href"),
                }
            )
        if len(out) >= max_records:
            break
    return out