"""Information extraction from free-text listings -> dense numeric features.

The competition hands us only unstructured sales copy, so everything a normal
property model would get as a column (beds, baths, sqft, lot size, year built)
has to be mined back out of the prose with regexes, plus amenity / condition /
property-type signals that move price.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1, "an": 1, "single": 1, "double": 2, "triple": 3,
}
NUM_WORD_RE = "|".join(WORD_NUM)


def _num(tok: str) -> float:
    tok = tok.strip().lower().replace(",", "")
    if tok in WORD_NUM:
        return float(WORD_NUM[tok])
    # fractions like "3/4"
    if "/" in tok:
        try:
            a, b = tok.split("/")
            return float(a) / float(b)
        except Exception:
            return np.nan
    try:
        return float(tok)
    except Exception:
        return np.nan


def _first(pattern: re.Pattern, text: str) -> float:
    m = pattern.search(text)
    return _num(m.group(1)) if m else np.nan


NUMTOK = rf"(\d+(?:[.,]\d+)?|\d+/\d+|{NUM_WORD_RE})"

RE_BED = re.compile(rf"{NUMTOK}[\s-]*(?:bed(?:room)?s?|br\b|bdrm)", re.I)
RE_BED2 = re.compile(rf"(?:bed(?:room)?s?)[:\s-]*{NUMTOK}", re.I)
RE_BATH = re.compile(rf"{NUMTOK}[\s-]*(?:full[\s-]*|half[\s-]*)?(?:bath(?:room)?s?|ba\b)", re.I)
RE_FULLBATH = re.compile(rf"{NUMTOK}[\s-]*full[\s-]*bath", re.I)
RE_HALFBATH = re.compile(rf"{NUMTOK}[\s-]*half[\s-]*bath", re.I)
RE_SQFT = re.compile(r"([\d,]{3,9}(?:\.\d+)?)\s*(?:\+/-\s*)?(?:sq\.?\s*(?:ft|feet)|square\s*(?:foot|feet)|sf\b)", re.I)
RE_ACRE = re.compile(rf"{NUMTOK}(?:\s*\+/-)?\s*(?:\+)?\s*acres?", re.I)
RE_ACRE_OF = re.compile(rf"{NUMTOK}\s*of\s*an?\s*acre", re.I)
RE_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")
RE_BUILT = re.compile(r"(?:built|constructed|circa|est\.?)\s*(?:in\s*)?(1[89]\d{2}|20[0-2]\d)", re.I)
RE_GARAGE = re.compile(rf"{NUMTOK}[\s-]*(?:car|bay)[\s-]*(?:attached\s*|detached\s*)?(?:garage|carport)", re.I)
RE_STORY = re.compile(rf"{NUMTOK}[\s-]*(?:story|storey|stories|level)", re.I)
RE_HOA = re.compile(r"hoa[^.]{0,40}?\$\s*([\d,]+)", re.I)
RE_TAX = re.compile(r"tax(?:es)?[^.]{0,40}?\$\s*([\d,]+)", re.I)
RE_RENT = re.compile(r"(?:rent|rents|rental|income)[^.]{0,40}?\$\s*([\d,]+)", re.I)
RE_DOLLAR = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*(k|m|million)?", re.I)
RE_UNITS = re.compile(rf"{NUMTOK}[\s-]*(?:unit|units|apartment|plex)", re.I)
RE_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")

# keyword -> feature name. Counted (not just flagged) because emphasis matters.
KEYWORDS = {
    "kw_luxury": r"luxur|prestigious|exquisite|opulent|magnificent|estate\b|bespoke|world-class",
    "kw_custom": r"custom[\s-]?(?:built|home|design)|architect",
    "kw_waterfront": r"waterfront|water\s*front|oceanfront|lakefront|riverfront|beachfront|bayfront",
    "kw_water_view": r"ocean|lake|river|beach|bay\b|waterview|water view|canal|dock|marina|boat\s*(?:slip|lift|house)",
    "kw_view": r"\bview(?:s)?\b|panoramic|vista|skyline|mountain view",
    "kw_pool": r"\bpool\b|spa\b|hot tub|jacuzzi",
    "kw_highend_finish": r"granite|quartz|marble|stainless|hardwood|travertine|chef'?s kitchen|gourmet|sub[\s-]?zero|wolf\b|viking\b|thermador|cathedral ceiling|coffered|crown moulding|crown molding",
    "kw_new_build": r"new construction|newly built|brand new|to be built|never lived|builder",
    "kw_renovated": r"renovat|remodel|updated|upgraded|refinish|restored|new roof|new furnace|new hvac|new windows|move[\s-]in ready|turn[\s-]?key",
    "kw_fixer": r"fixer|handyman|tlc\b|as[\s-]is|needs work|investor special|rehab|distressed|foreclos|short sale|auction|sold as|cash only|gut\b",
    "kw_condo": r"\bcondo|condominium|co-?op\b",
    "kw_townhouse": r"townh(?:ouse|ome)|row house|duplex|end[\s-]unit",
    "kw_mobile": r"mobile home|manufactur(?:ed)? home|doublewide|double[\s-]wide|singlewide|trailer|modular",
    "kw_land": r"\bland\b|vacant lot|buildable|raw land|parcel|acreage|timber|pasture|farmland|ranch land",
    "kw_multifamily": r"multi[\s-]?family|duplex|triplex|fourplex|apartment building|income property|cap rate|noi\b|tenant",
    "kw_commercial": r"commercial|retail|office space|warehouse|industrial|zoned c|business",
    "kw_hoa": r"\bhoa\b|association fee|condo fee|maintenance fee",
    "kw_gated": r"gated|guard[\s-]?house|concierge|doorman|private community|country club|golf",
    "kw_garage": r"garage|carport",
    "kw_basement": r"basement|cellar|lower level",
    "kw_fireplace": r"fireplace|wood stove|pellet stove",
    "kw_deck": r"\bdeck\b|patio|porch|lanai|veranda|terrace|courtyard",
    "kw_elevator": r"elevator|dumbwaiter",
    "kw_guest": r"guest (?:house|suite|cottage)|casita|in[\s-]?law|adu\b|carriage house|accessory dwelling",
    "kw_wine": r"wine (?:cellar|room)|theater|theatre|media room|home gym|sauna|bowling",
    "kw_solar": r"solar|geothermal|energy efficient|net zero|ev charg",
    "kw_barn": r"\bbarn\b|stable|equestrian|paddock|corral|silo|chicken",
    "kw_school": r"school|district|university|college|campus",
    "kw_commute": r"commut|highway|freeway|interstate|metro|subway|train station|airport|minutes (?:from|to)",
    "kw_downtown": r"downtown|city center|walk(?:able|ing distance)|shops|dining|restaurants",
    "kw_historic": r"historic|victorian|colonial|craftsman|tudor|antique|century|heritage|landmark",
    "kw_modern": r"modern|contemporary|mid[\s-]century|minimalist|sleek|open concept|smart home",
    "kw_ranch_style": r"ranch[\s-]?style|\branch\b|bungalow|cape cod|split[\s-]level|raised ranch|cottage|farmhouse",
    "kw_penthouse": r"penthouse|high[\s-]rise|doorman|loft\b",
    "kw_acre_word": r"acre",
    "kw_privacy": r"private|seclu|tranquil|serene|retreat|oasis|sanctuary|wooded|cul[\s-]de[\s-]sac",
    "kw_rare": r"rare(?:ly)?|one of a kind|unique|opportunity|potential|must see|won'?t last|priced to sell|motivated",
    "kw_starter": r"starter home|first[\s-]time|affordable|budget|value|bargain",
    "kw_sqft_word": r"square (?:foot|feet)|sq\.? ?ft",
}
KEYWORD_RE = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}

US_STATES = (
    "alabama alaska arizona arkansas california colorado connecticut delaware florida georgia "
    "hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts "
    "michigan minnesota mississippi missouri montana nebraska nevada hampshire jersey mexico "
    "york carolina dakota ohio oklahoma oregon pennsylvania rhode tennessee texas utah vermont "
    "virginia washington wisconsin wyoming"
).split()
STATE_RE = re.compile(r"\b(" + "|".join(US_STATES) + r")\b", re.I)


def _dollar_amounts(text: str) -> list:
    out = []
    for amt, suffix in RE_DOLLAR.findall(text):
        v = _num(amt)
        if np.isnan(v):
            continue
        s = (suffix or "").lower()
        if s == "k":
            v *= 1e3
        elif s in ("m", "million"):
            v *= 1e6
        out.append(v)
    return out


def extract_row(text: str) -> dict:
    t = text if isinstance(text, str) else ""
    low = t.lower()
    f = {}

    beds = _first(RE_BED, t)
    if np.isnan(beds):
        beds = _first(RE_BED2, t)
    f["beds"] = beds if (not np.isnan(beds) and 0 < beds <= 30) else np.nan

    baths = _first(RE_BATH, t)
    f["baths"] = baths if (not np.isnan(baths) and 0 < baths <= 30) else np.nan
    f["full_baths"] = _first(RE_FULLBATH, t)
    f["half_baths"] = _first(RE_HALFBATH, t)

    sqft_vals = [_num(x) for x in RE_SQFT.findall(t)]
    sqft_vals = [v for v in sqft_vals if not np.isnan(v) and 100 <= v <= 60000]
    f["sqft"] = max(sqft_vals) if sqft_vals else np.nan
    f["sqft_min"] = min(sqft_vals) if sqft_vals else np.nan
    f["n_sqft_mentions"] = len(sqft_vals)

    acres = [_num(x) for x in RE_ACRE.findall(t)] + [_num(x) for x in RE_ACRE_OF.findall(t)]
    acres = [v for v in acres if not np.isnan(v) and 0 < v <= 100000]
    f["acres"] = max(acres) if acres else np.nan

    yb = _first(RE_BUILT, t)
    years = [float(y) for y in RE_YEAR.findall(t)]
    f["year_built"] = yb if not np.isnan(yb) else (min(years) if years else np.nan)
    f["year_recent"] = max(years) if years else np.nan
    f["n_years"] = len(years)

    f["garage_cars"] = _first(RE_GARAGE, t)
    f["stories"] = _first(RE_STORY, t)
    f["units"] = _first(RE_UNITS, t)
    f["hoa_fee"] = _first(RE_HOA, t)
    f["tax_amt"] = _first(RE_TAX, t)
    f["rent_amt"] = _first(RE_RENT, t)

    amounts = _dollar_amounts(t)
    f["n_dollar"] = len(amounts)
    f["max_dollar"] = max(amounts) if amounts else np.nan
    f["min_dollar"] = min(amounts) if amounts else np.nan

    nums = [float(x.replace(",", "")) for x in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", t)[:200]]
    nums = [n for n in nums if n < 1e9]
    f["n_numbers"] = len(nums)
    f["max_number"] = max(nums) if nums else np.nan
    f["median_number"] = float(np.median(nums)) if nums else np.nan

    f["len_chars"] = len(t)
    words = low.split()
    f["len_words"] = len(words)
    f["n_sentences"] = t.count(".") + t.count("!") + t.count("?")
    f["avg_word_len"] = float(np.mean([len(w) for w in words])) if words else 0.0
    f["n_upper_words"] = sum(1 for w in t.split() if w.isupper() and len(w) > 2)
    f["upper_ratio"] = sum(1 for c in t if c.isupper()) / max(len(t), 1)
    f["digit_ratio"] = sum(1 for c in t if c.isdigit()) / max(len(t), 1)
    f["excl_marks"] = t.count("!")
    f["n_commas"] = t.count(",")
    f["n_redacted"] = low.count("[redacted")
    f["has_zip"] = 1.0 if RE_ZIP.search(t) else 0.0
    f["n_states"] = len(set(m.lower() for m in STATE_RE.findall(t)))

    for name, rx in KEYWORD_RE.items():
        f[name] = float(len(rx.findall(t)))

    # combinations that a structured model would have had
    f["bed_bath"] = f["beds"] * f["baths"]
    f["sqft_per_bed"] = f["sqft"] / f["beds"] if (not np.isnan(f["sqft"]) and not np.isnan(f["beds"]) and f["beds"] > 0) else np.nan
    f["log_sqft"] = np.log1p(f["sqft"]) if not np.isnan(f["sqft"]) else np.nan
    f["log_acres"] = np.log1p(f["acres"]) if not np.isnan(f["acres"]) else np.nan
    f["log_max_dollar"] = np.log1p(f["max_dollar"]) if not np.isnan(f["max_dollar"]) else np.nan
    f["age"] = 2025 - f["year_built"] if not np.isnan(f["year_built"]) else np.nan
    return f


def build_features(texts) -> pd.DataFrame:
    rows = [extract_row(t) for t in texts]
    df = pd.DataFrame(rows).astype(np.float32)
    return df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import load_data, to_log

    train, test = load_data()
    feats = build_features(train["text"])
    y = to_log(train["listPrice"].values)
    corr = feats.apply(lambda c: pd.Series(c).corr(pd.Series(y)))
    print(feats.shape, "non-null rate / corr with log price:")
    summary = pd.DataFrame({"nonnull": feats.notna().mean().round(3), "corr": corr.round(3)})
    print(summary.sort_values("corr", key=abs, ascending=False).head(40).to_string())
