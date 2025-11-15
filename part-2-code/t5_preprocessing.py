#!/usr/bin/env python
import os
import json
import re
from typing import List, Tuple, Dict, Set

from transformers import T5TokenizerFast


# ---------- CONFIG ----------

DATA_DIR = "data"
# Make sure this meta file is in the same directory
SCHEMA_META_PATH = "schema_meta.json" 


# ---------- BASIC TEXT HELPERS ----------

def normalize_space(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return " ".join(text.strip().split())


def normalize_lower(text: str) -> str:
    """Lowercase + collapse whitespace (for matching)."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------- LOAD SCHEMA META ----------
# This block runs when the file is imported

try:
    with open(SCHEMA_META_PATH, "r", encoding="utf-8") as f:
        SCHEMA_META = json.load(f)

    ENTS: Dict[str, Dict] = SCHEMA_META["ents"]
    DEFAULTS: Dict[str, Dict] = SCHEMA_META["defaults"]
    LINKS: Dict[str, Dict] = SCHEMA_META["links"]
except FileNotFoundError:
    print(f"Error: '{SCHEMA_META_PATH}' not found.")
    print("Please make sure your schema metadata file is in the correct location.")
    # We define them as empty dicts so the import doesn't crash,
    # but the script will fail later.
    ENTS, DEFAULTS, LINKS = {}, {}, {}


# ---------- BUILD PHRASE → TABLE LEXICON ----------

PHRASE2TABLE: Dict[str, Set[str]] = {}

# table default utterances
for table, meta in DEFAULTS.items():
    phrase = normalize_lower(meta["utt"])
    PHRASE2TABLE.setdefault(phrase, set()).add(table)

# column utterances
for table, cols in ENTS.items():
    for col, colmeta in cols.items():
        phrase = normalize_lower(colmeta["utt"])
        PHRASE2TABLE.setdefault(phrase, set()).add(table)


# ---------- QUESTION-CONDITIONED SCHEMA SELECTION ----------

def detect_tables_for_question(question: str, max_tables: int = 8) -> Set[str]:
    """
    Heuristically select a small set of relevant tables for a question.
    """
    q = normalize_lower(question)
    candidates: Set[str] = set()

    # step 1: direct matches
    for phrase, tables in PHRASE2TABLE.items():
        if phrase and phrase in q:
            candidates.update(tables)

    # fallback: if no match, be conservative and use all tables
    if not candidates:
        return set(ENTS.keys())

    # step 2: 1-hop neighbor expansion
    expanded: Set[str] = set(candidates)

    # forward neighbors
    for t in list(candidates):
        for neigh, fk_col in LINKS.get(t, {}).items():
            expanded.add(neigh)

    # reverse neighbors (other tables pointing to t)
    for other, neighs in LINKS.items():
        for neigh, fk_col in neighs.items():
            if neigh in candidates:
                expanded.add(other)

    # step 3: cap number of tables
    if len(expanded) > max_tables:
        # keep all original candidates if possible; trim neighbors
        base = list(candidates)
        # ensure deterministic order
        base = sorted(base)
        expanded = set(base[:max_tables])

    return expanded


def serialize_tables(tables: Set[str], max_cols_per_table: int = 6) -> str:
    """
    Serialize selected tables into a compact 'table(col1, col2, ...)' format.
    """
    parts = []
    # Sort for deterministic order
    for t in sorted(tables):
        # Ensure we don't crash if a bad table name is passed
        if t not in ENTS:
            continue
            
        cols = list(ENTS[t].keys())
        cols = cols[:max_cols_per_table]
        col_list = ", ".join(cols)
        parts.append(f"{t}({col_list})")
        
    return "Tables:\n" + ",\n".join(parts)


# ---------- ENCODER / DECODER PREPROCESSING ----------

def build_encoder_input(nl_question: str) -> str:
    """
    Build the actual text fed to the T5 encoder:
    task instruction + question-conditioned schema + NL question.
    """
    q = normalize_space(nl_question)
    candidate_tables = detect_tables_for_question(q)
    schema_str = serialize_tables(candidate_tables)

    # This is the full prompt for the encoder
    return (
        "translate the following question into a SQL query over the flight database.\n"
        f"{schema_str}\n\n"
        f"Question: {q}\n"
        "SQL:"
    )


def build_decoder_target(sql: str) -> str:
    """
    Canonical SQL format for the decoder target.
    - Normalize whitespace.
    - Uppercase keywords and identifiers for consistency.
    """
    s = normalize_space(sql)
    # Convert to uppercase as defined in your strategy
    s = s.upper() 
    return s


# ---------- IO & STATS HELPERS ----------

def read_lines(path: str) -> List[str]:
    """
    Helper function to read data files.
    This is imported by load_data.py
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]