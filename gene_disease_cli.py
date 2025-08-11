#!/usr/bin/env python3
"""
Gene–disease CLI — analyze potential correlations using public life sciences data + an optional LLM.

Data sources (no API keys required):
- Ensembl REST: gene symbol → Ensembl ID + synonyms
- EBI OLS4: disease name → EFO/MONDO ID + synonyms
- Open Targets Platform GraphQL: association + datatype scores for target–disease pair
- Europe PMC REST: recent literature snippets mentioning both the gene and disease
- (Optional) GWAS Catalog REST: associations for the EFO trait, filtered to the gene

LLM (optional):
- OpenAI Chat Completions API (set OPENAI_API_KEY; choose model with --model / -m)

Quickstart with uv:
    uv run --with typer --with httpx --with tenacity --with pydantic \
      gene_disease_cli.py analyze --gene IL22 --disease psoriasis --since 2015 --max-abstracts 8 --model gpt-4o-mini

History / show / export:
    uv run --with typer --with httpx --with tenacity --with pydantic gene_disease_cli.py history
    uv run --with typer --with httpx --with tenacity --with pydantic gene_disease_cli.py show 1
    uv run --with typer --with httpx --with tenacity --with pydantic gene_disease_cli.py export 1 --out run1.json

Notes:
- Results + raw evidence + prompts are stored in ~/.gene_disease_cli/history.db
- Compatible with older OpenAI models (auto-retries without response_format when needed).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import typer
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

APP = typer.Typer(
    help="Analyze potential correlations between genes and diseases using public data + LLM."
)
DB_PATH = os.path.expanduser("~/.gene_disease_cli/history.db")
OT_GQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
ENSEMBL_LOOKUP = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}?content-type=application/json"
OLS_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api"

# ----------------------------
# Utilities
# ----------------------------


def ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS genes(
        id INTEGER PRIMARY KEY,
        symbol TEXT,
        ensembl_id TEXT,
        synonyms_json TEXT
    );
    """
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS diseases(
        id INTEGER PRIMARY KEY,
        label TEXT,
        efo_id TEXT,
        mondo_id TEXT,
        synonyms_json TEXT
    );
    """
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY,
        ts TEXT,
        gene_symbol TEXT,
        ensembl_id TEXT,
        disease_label TEXT,
        disease_efo TEXT,
        params_json TEXT,
        model TEXT
    );
    """
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS evidence(
        id INTEGER PRIMARY KEY,
        run_id INTEGER,
        source TEXT,
        payload_json TEXT
    );
    """
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS analysis(
        run_id INTEGER PRIMARY KEY,
        verdict TEXT,
        confidence REAL,
        rationale TEXT,
        output_json TEXT
    );
    """
    )
    con.commit()
    con.close()


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


# ----------------------------
# Data model passed to LLM
# ----------------------------


class LiteEvidence(BaseModel):
    query: Dict[str, Any]
    synonyms: Dict[str, List[str]] = Field(default_factory=dict)
    opentargets: Optional[Dict[str, Any]] = None
    gwas_catalog: Optional[List[Dict[str, Any]]] = None
    literature: List[Dict[str, Any]] = Field(default_factory=list)


# ----------------------------
# HTTP helpers
# ----------------------------

HEADERS_JSON = {"Accept": "application/json"}
HEADERS_GQL = {"Content-Type": "application/json", "Accept": "application/json"}


@retry(wait=wait_exponential(multiplier=0.5, min=1, max=8), stop=stop_after_attempt(3))
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
        resp = await client.get(url, params=params, headers=headers, timeout=30)
    else:
        resp = await client.post(
            url, params=params, headers=headers, json=json_body, timeout=45
        )
    resp.raise_for_status()
    return resp.json()


# ----------------------------
# Resolvers (IDs + synonyms)
# ----------------------------


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
        "rows": 25,
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


# ----------------------------
# Open Targets (association + datatype scores)
# ----------------------------

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


async def ot_find_association(
    client: httpx.AsyncClient, ensg: str, efo_id: str
) -> Optional[Dict[str, Any]]:
    index = 0
    size = 50
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


# ----------------------------
# Europe PMC (literature snippets)
# ----------------------------


async def europe_pmc_literature(
    client: httpx.AsyncClient,
    gene_terms: List[str],
    disease_terms: List[str],
    since_year: int,
    max_records: int,
) -> List[Dict[str, Any]]:
    gene_q = " OR ".join([f'"{t}"' if " " in t else t for t in gene_terms[:5]])
    dis_q = " OR ".join([f'"{t}"' if " " in t else t for t in disease_terms[:8]])
    query = f"({gene_q}) AND ({dis_q}) AND (PUB_YEAR:[{since_year} TO 3000])"
    params = {
        "query": query,
        "resultType": "core",
        "pageSize": min(max_records, 25),
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


# ----------------------------
# GWAS Catalog (by EFO trait, filtered to gene)
# ----------------------------


async def gwas_associations_by_efo_for_gene(
    client: httpx.AsyncClient,
    efo_label_or_id: str,
    gene_symbol: str,
    max_records: int = 20,
) -> List[Dict[str, Any]]:
    params = {"efoTrait": efo_label_or_id, "size": min(max_records, 1000)}
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


# ----------------------------
# LLM (OpenAI Chat Completions) — optional
# ----------------------------

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


async def call_openai_chat(
    evidence_json: Dict[str, Any], model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Task: Evaluate correlation between {gene} and {disease}. Return JSON only. Use the provided evidence bundle.\n\nEVIDENCE:\n"
                + json.dumps(evidence_json, ensure_ascii=False),
            },
        ],
    }
    async with httpx.AsyncClient(timeout=90) as client:
        try:
            # First try with JSON response_format (newer models)
            payload_with_json = dict(payload, response_format={"type": "json_object"})
            resp = await client.post(
                f"{base_url}/chat/completions", headers=headers, json=payload_with_json
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            txt = e.response.text or ""
            # Older models (e.g., gpt-4-0613) don’t support response_format — retry without it
            if e.response.status_code == 400 and "response_format" in txt.lower():
                resp = await client.post(
                    f"{base_url}/chat/completions", headers=headers, json=payload
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
    except Exception:
        # Best effort if the model returned plain text
        return {
            "verdict": "inconclusive",
            "confidence": 0.0,
            "drivers": {},
            "key_points": [],
            "conflicts_or_gaps": [],
            "recommended_next_steps": [],
            "_raw": content,
        }


# ----------------------------
# Main analysis flow
# ----------------------------


@dataclass
class AnalyzeParams:
    gene: str
    disease: str
    since: int = 2015
    max_abstracts: int = 8
    include_gwas: bool = True
    model: Optional[str] = None  # if None, skip LLM and just store evidence


async def analyze_once(
    params: AnalyzeParams,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    ensure_db()

    async with httpx.AsyncClient() as client:
        ensg, gene_syns = await resolve_gene_symbol(client, params.gene)
        efo_id, mondo_id, dis_syns = await resolve_disease_label(client, params.disease)
        ot_assoc = await ot_find_association(client, ensg, efo_id)
        literature = await europe_pmc_literature(
            client, gene_syns, dis_syns, params.since, params.max_abstracts
        )
        gwas = (
            await gwas_associations_by_efo_for_gene(
                client,
                params.disease if "EFO_" not in params.disease else params.disease,
                params.gene,
                max_records=15,
            )
            if params.include_gwas
            else []
        )

    evidence = LiteEvidence(
        query={
            "gene": {"symbol": params.gene, "ensembl_id": ensg},
            "disease": {"label": params.disease, "efo_id": efo_id},
        },
        synonyms={"gene": gene_syns, "disease": dis_syns},
        opentargets=ot_assoc,
        gwas_catalog=gwas,
        literature=literature,
    ).model_dump()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO runs(ts,gene_symbol,ensembl_id,disease_label,disease_efo,params_json,model) VALUES(?,?,?,?,?,?,?)",
        (
            now_iso(),
            params.gene,
            ensg,
            params.disease,
            efo_id,
            json.dumps(asdict(params)),
            params.model or "",
        ),
    )
    run_id = cur.lastrowid
    cur.execute(
        "INSERT INTO evidence(run_id, source, payload_json) VALUES(?,?,?)",
        (run_id, "bundle", json.dumps(evidence)),
    )
    con.commit()

    llm_output = None
    if params.model:
        try:
            llm_output = await call_openai_chat(evidence, model=params.model)
            cur.execute(
                "INSERT OR REPLACE INTO analysis(run_id, verdict, confidence, rationale, output_json) VALUES(?,?,?,?,?)",
                (
                    run_id,
                    llm_output.get("verdict", "inconclusive"),
                    float(llm_output.get("confidence", 0.0) or 0.0),
                    json.dumps(llm_output.get("key_points", [])),
                    json.dumps(llm_output),
                ),
            )
            con.commit()
        except Exception as e:
            err = {"error": f"{e.__class__.__name__}: {e}"}
            cur.execute(
                "INSERT OR REPLACE INTO analysis(run_id, verdict, confidence, rationale, output_json) VALUES(?,?,?,?,?)",
                (run_id, "inconclusive", 0.0, f"LLM error: {e}", json.dumps(err)),
            )
            con.commit()
            llm_output = err
    con.close()

    return evidence, llm_output


# ----------------------------
# CLI commands
# ----------------------------


@APP.command()
def analyze(
    gene: str = typer.Option(..., help="Gene symbol (e.g., IL22)"),
    disease: str = typer.Option(
        ..., help="Disease name or EFO/MONDO ID (e.g., psoriasis)"
    ),
    since: int = typer.Option(2015, help="Minimum publication year for literature"),
    max_abstracts: int = typer.Option(
        8, min=1, max=25, help="Max literature items to include"
    ),
    include_gwas: bool = typer.Option(
        True, help="Try to fetch GWAS Catalog associations (best-effort)"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model for analysis (OpenAI). If omitted, skip LLM step.",
    ),
):
    """Run one analysis. Saves a history entry and prints a concise summary."""
    ev, out = asyncio.run(
        analyze_once(
            AnalyzeParams(
                gene=gene.strip(),
                disease=disease.strip(),
                since=since,
                max_abstracts=max_abstracts,
                include_gwas=include_gwas,
                model=model,
            )
        )
    )
    typer.secho("== Normalized inputs ==", fg=typer.colors.CYAN, bold=True)
    print(json.dumps(ev["query"], indent=2))
    if ev.get("opentargets"):
        typer.secho(
            "\n== Open Targets association (overall + datatype scores) ==",
            fg=typer.colors.CYAN,
            bold=True,
        )
        print(json.dumps(ev["opentargets"], indent=2))
    if ev.get("gwas_catalog"):
        typer.secho(
            "\n== GWAS Catalog (filtered to gene) ==", fg=typer.colors.CYAN, bold=True
        )
        print(json.dumps(ev["gwas_catalog"][:5], indent=2))
    if ev.get("literature"):
        typer.secho(
            "\n== Literature snippets (Europe PMC) ==", fg=typer.colors.CYAN, bold=True
        )
        for lit in ev["literature"][:3]:
            print(f"- PMID {lit.get('pmid')} ({lit.get('year')}): {lit.get('title')}")
            for s in lit.get("sentences", []):
                print(f"  • {s}")
    if out:
        typer.secho("\n== LLM verdict ==", fg=typer.colors.GREEN, bold=True)
        print(json.dumps(out, indent=2))
    else:
        typer.secho(
            "\n(No LLM model specified; stored evidence bundle only.)",
            fg=typer.colors.YELLOW,
        )


@APP.command()
def history():
    """List prior runs."""
    ensure_db()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT id, ts, gene_symbol, disease_label, model FROM runs ORDER BY id DESC LIMIT 50"
    )
    rows = cur.fetchall()
    con.close()
    if not rows:
        typer.echo("No runs yet.")
        raise typer.Exit()
    for r in rows:
        print(f"[{r[0]}] {r[1]}  {r[2]} ↔ {r[3]}  (model: {r[4] or '—'})")


@APP.command()
def show(run_id: int):
    """Show stored LLM output (if any) and a compact summary for a run."""
    ensure_db()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT payload_json FROM evidence WHERE run_id=? AND source='bundle'",
        (run_id,),
    )
    evrow = cur.fetchone()
    cur.execute("SELECT output_json FROM analysis WHERE run_id=?", (run_id,))
    anrow = cur.fetchone()
    con.close()
    if not evrow:
        typer.echo(f"No evidence found for run {run_id}.")
        raise typer.Exit(code=1)
    evidence = json.loads(evrow[0])
    typer.secho("== Inputs ==", fg=typer.colors.CYAN, bold=True)
    print(json.dumps(evidence["query"], indent=2))
    if anrow:
        typer.secho("\n== LLM output ==", fg=typer.colors.GREEN, bold=True)
        print(json.dumps(json.loads(anrow[0]), indent=2))
    else:
        typer.secho("\n(No LLM output stored for this run.)", fg=typer.colors.YELLOW)


@APP.command()
def export(
    run_id: int,
    out: Path = typer.Option(
        ..., exists=False, dir_okay=False, writable=True, help="Destination JSON file"
    ),
):
    """Export the evidence + analysis for a run as JSON."""
    ensure_db()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT payload_json FROM evidence WHERE run_id=? AND source='bundle'",
        (run_id,),
    )
    evrow = cur.fetchone()
    cur.execute("SELECT output_json FROM analysis WHERE run_id=?", (run_id,))
    anrow = cur.fetchone()
    con.close()
    if not evrow:
        typer.echo(f"No evidence found for run {run_id}.")
        raise typer.Exit(code=1)
    out_data = {
        "evidence": json.loads(evrow[0]),
        "analysis": json.loads(anrow[0]) if anrow else None,
    }
    out.write_text(json.dumps(out_data, indent=2))
    typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    APP()
