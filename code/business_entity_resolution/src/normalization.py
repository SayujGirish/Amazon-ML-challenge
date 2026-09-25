import re
import unicodedata
import pandas as pd


# ---------------------------------------------------------
# Basic text normalization
# ---------------------------------------------------------

def normalize_text(text):
    """
    General text normalization.

    - Handles missing values
    - Standardizes Unicode
    - Converts to lowercase
    - Replaces '&' with 'and'
    - Removes punctuation
    - Normalizes whitespace
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Treat "&" and "and" consistently
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Advanced business-name normalization
# ---------------------------------------------------------

def normalize_business_name(name):
    """
    Normalize business names.

    Includes:
    - Basic text normalization
    - Common legal/business abbreviations
    """

    name = normalize_text(name)

    if not name:
        return ""

    # Common business/legal abbreviations
    replacements = {
        r"\bpvt\b": "private",
        r"\bpty\b": "proprietary",
        r"\bltd\b": "limited",
        r"\bcorp\b": "corporation",
        r"\bco\b": "company",
        r"\binc\b": "incorporated",
    }

    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)

    # Normalize whitespace again
    name = re.sub(r"\s+", " ", name).strip()

    return name


# ---------------------------------------------------------
# Advanced address normalization
# ---------------------------------------------------------

def normalize_business_address(address):
    """
    Normalize business addresses.

    Includes:
    - Basic text normalization
    - Common address abbreviations
    """

    address = normalize_text(address)

    if not address:
        return ""

    # Common address abbreviations
    replacements = {
        r"\brd\b": "road",
        r"\broad\b": "road",

        r"\bst\b": "street",
        r"\bstreet\b": "street",

        r"\bave\b": "avenue",
        r"\bav\b": "avenue",
        r"\baven\b": "avenue",
        r"\bavenue\b": "avenue",

        r"\bblvd\b": "boulevard",
        r"\bblvd\b": "boulevard",

        r"\bln\b": "lane",
        r"\blane\b": "lane",

        r"\bdr\b": "drive",
        r"\bdrive\b": "drive",

        r"\bhwy\b": "highway",
        r"\bhighway\b": "highway",

        r"\bapt\b": "apartment",
        r"\bapartment\b": "apartment",

        r"\bsuite\b": "suite",
        r"\bste\b": "suite",
    }

    for pattern, replacement in replacements.items():
        address = re.sub(pattern, replacement, address)

    # Normalize whitespace
    address = re.sub(r"\s+", " ", address).strip()

    return address


# ---------------------------------------------------------
# Country normalization
# ---------------------------------------------------------

def normalize_country(country):
    """
    Normalize country names.

    - Handles missing values
    - Standardizes Unicode
    - Converts to lowercase
    - Normalizes whitespace
    """

    if pd.isna(country):
        return ""

    country = str(country)

    country = unicodedata.normalize("NFKC", country)

    country = country.lower()

    country = re.sub(r"\s+", " ", country).strip()

    return country


# ---------------------------------------------------------
# Apply normalization to one source
# ---------------------------------------------------------

def normalize_source(df):
    """
    Keep original columns and add normalized columns.
    """

    df = df.copy()

    df["business_name_normalized"] = (
        df["business_name"].apply(normalize_business_name)
    )

    df["business_address_normalized"] = (
        df["business_address"].apply(normalize_business_address)
    )

    df["country_normalized"] = (
        df["country"].apply(normalize_country)
    )

    return df


# ---------------------------------------------------------
# Normalize all three sources
# ---------------------------------------------------------

def normalize_all_sources(source1, source2, source3):

    source1 = normalize_source(source1)
    source2 = normalize_source(source2)
    source3 = normalize_source(source3)

    return source1, source2, source3