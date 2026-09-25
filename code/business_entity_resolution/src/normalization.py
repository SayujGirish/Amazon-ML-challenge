import re
import unicodedata
import pandas as pd


# ---------------------------------------------------------
# LIGHT NORMALIZATION
# Used before blocking
# ---------------------------------------------------------

def light_normalize(value):
    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Replace & with and
    value = value.replace("&", " and ")

    # Remove punctuation
    value = re.sub(r"[^\w\s]", " ", value)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def light_normalize_source(df):
    df = df.copy()

    df["business_name_normalized"] = (
        df["business_name"].map(light_normalize)
    )

    df["business_address_normalized"] = (
        df["business_address"].map(light_normalize)
    )

    df["country_normalized"] = (
        df["country"].map(light_normalize)
    )

    return df


# ---------------------------------------------------------
# ADVANCED NORMALIZATION
# Used later on candidate records
# ---------------------------------------------------------

BUSINESS_REPLACEMENTS = {
    r"\bpvt\b": "private",
    r"\bpty\b": "proprietary",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\binc\b": "incorporated",
}

ADDRESS_REPLACEMENTS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\baven\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bhwy\b": "highway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
}


def advanced_normalize_business_name(value):
    value = light_normalize(value)

    for pattern, replacement in BUSINESS_REPLACEMENTS.items():
        value = re.sub(pattern, replacement, value)

    return value


def advanced_normalize_address(value):
    value = light_normalize(value)

    for pattern, replacement in ADDRESS_REPLACEMENTS.items():
        value = re.sub(pattern, replacement, value)

    return value


def advanced_normalize_source(df):
    df = df.copy()

    df["business_name_normalized"] = (
        df["business_name"].map(advanced_normalize_business_name)
    )

    df["business_address_normalized"] = (
        df["business_address"].map(advanced_normalize_address)
    )

    df["country_normalized"] = (
        df["country"].map(light_normalize)
    )

    return df