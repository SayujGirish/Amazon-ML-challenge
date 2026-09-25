import pandas as pd
from rapidfuzz import fuzz


# ---------------------------------------------------------
# Basic similarity functions
# ---------------------------------------------------------

def exact_match(value1, value2):
    """
    Return 1 if both values are exactly equal and non-empty.
    Otherwise return 0.
    """

    if not value1 or not value2:
        return 0

    return int(value1 == value2)


def similarity_ratio(value1, value2):
    """
    Standard fuzzy string similarity.

    Returns a value between 0 and 1.
    """

    if not value1 or not value2:
        return 0.0

    return fuzz.ratio(value1, value2) / 100.0


def token_sort_similarity(value1, value2):
    """
    Compare strings while ignoring word order.

    Example:
        "abc global technologies"
        "technologies abc global"

    can receive a high similarity score.
    """

    if not value1 or not value2:
        return 0.0

    return fuzz.token_sort_ratio(value1, value2) / 100.0


def token_set_similarity(value1, value2):
    """
    Compare the sets of words in two strings.

    Useful when one string contains additional words.
    """

    if not value1 or not value2:
        return 0.0

    return fuzz.token_set_ratio(value1, value2) / 100.0


# ---------------------------------------------------------
# Calculate features for one candidate pair
# ---------------------------------------------------------

def calculate_pair_features(s1, candidate):
    """
    Calculate similarity features between one Source 1
    record and one Source 2/3 candidate.
    """

    s1_name = s1["business_name_normalized"]
    candidate_name = candidate["business_name_normalized"]

    s1_address = s1["business_address_normalized"]
    candidate_address = candidate["business_address_normalized"]

    s1_country = s1["country_normalized"]
    candidate_country = candidate["country_normalized"]

    features = {

        # -------------------------------------------------
        # Business name features
        # -------------------------------------------------

        "name_exact": exact_match(
            s1_name,
            candidate_name
        ),

        "name_similarity": similarity_ratio(
            s1_name,
            candidate_name
        ),

        "name_token_sort_similarity": token_sort_similarity(
            s1_name,
            candidate_name
        ),

        "name_token_set_similarity": token_set_similarity(
            s1_name,
            candidate_name
        ),

        # -------------------------------------------------
        # Address features
        # -------------------------------------------------

        "address_exact": exact_match(
            s1_address,
            candidate_address
        ),

        "address_similarity": similarity_ratio(
            s1_address,
            candidate_address
        ),

        "address_token_sort_similarity": token_sort_similarity(
            s1_address,
            candidate_address
        ),

        "address_token_set_similarity": token_set_similarity(
            s1_address,
            candidate_address
        ),

        # -------------------------------------------------
        # Country feature
        # -------------------------------------------------

        "country_exact": exact_match(
            s1_country,
            candidate_country
        ),
    }

    return features


# ---------------------------------------------------------
# Generate features for all candidate pairs
# ---------------------------------------------------------

def generate_features(candidate_pairs, source1, source2, source3):
    """
    Generate similarity features for every candidate pair.
    """

    # Create lookup dictionaries for fast access
    source1_lookup = source1.set_index(
        "entity_id"
    ).to_dict("index")

    source2_lookup = source2.set_index(
        "entity_id"
    ).to_dict("index")

    source3_lookup = source3.set_index(
        "entity_id"
    ).to_dict("index")

    feature_rows = []

    for _, pair in candidate_pairs.iterrows():

        s1_id = pair["source1_id"]
        candidate_id = pair["candidate_id"]
        source = pair["source"]

        # Get Source 1 record
        s1 = source1_lookup[s1_id]

        # Get candidate from Source 2 or Source 3
        if source == "source2":
            candidate = source2_lookup[candidate_id]
        else:
            candidate = source3_lookup[candidate_id]

        # Calculate similarity features
        features = calculate_pair_features(
            s1,
            candidate
        )

        # Keep pair identification information
        features["source1_id"] = s1_id
        features["candidate_id"] = candidate_id
        features["source"] = source

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)