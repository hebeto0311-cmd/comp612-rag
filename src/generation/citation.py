"""Check citation identity, not semantic entailment."""
import re

CITATION = re.compile(r'\[([A-Za-z0-9]+_\d{4})\]')


def extract_citations(text):
    return list(dict.fromkeys(CITATION.findall(text)))


def validate_citations(text, valid_ids):
    used = extract_citations(text)
    return used, [cid for cid in used if cid not in valid_ids]


def remove_citations(text, bad_ids):
    bad = set(bad_ids)
    return CITATION.sub(lambda m: '' if m.group(1) in bad else m.group(0), text)
