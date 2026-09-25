import pandas as pd


def create_block_key(row):
    """
    Create a blocking key using:
    - country
    - first few characters of normalized business name
    """

    country = row["country_normalized"]
    name = row["business_name_normalized"]

    # Take first 3 characters of business name
    name_prefix = name[:3]

    return f"{country}_{name_prefix}"


def create_blocks(source1, source2, source3):
    """
    Create blocks for all three sources.
    """

    source1 = source1.copy()
    source2 = source2.copy()
    source3 = source3.copy()

    # Create blocking keys
    source1["block_key"] = source1.apply(create_block_key, axis=1)
    source2["block_key"] = source2.apply(create_block_key, axis=1)
    source3["block_key"] = source3.apply(create_block_key, axis=1)

    return source1, source2, source3


def generate_candidate_pairs(source1, source2, source3):
    """
    Generate candidate pairs between:

    Source 1 ↔ Source 2
    Source 1 ↔ Source 3

    Only records belonging to the same block are compared.
    """

    pairs = []

    # Group Source 2 and Source 3 by block
    source2_blocks = {
        key: group
        for key, group in source2.groupby("block_key")
    }

    source3_blocks = {
        key: group
        for key, group in source3.groupby("block_key")
    }

    # Go through every Source 1 entity
    for _, s1 in source1.iterrows():

        block_key = s1["block_key"]

        # -----------------------------------------
        # Source 1 ↔ Source 2
        # -----------------------------------------

        if block_key in source2_blocks:

            candidates = source2_blocks[block_key]

            for _, s2 in candidates.iterrows():

                pairs.append({
                    "source1_id": s1["entity_id"],
                    "source2_id": s2["entity_id"],
                    "source": "source2",
                    "block_key": block_key
                })

        # -----------------------------------------
        # Source 1 ↔ Source 3
        # -----------------------------------------

        if block_key in source3_blocks:

            candidates = source3_blocks[block_key]

            for _, s3 in candidates.iterrows():

                pairs.append({
                    "source1_id": s1["entity_id"],
                    "source3_id": s3["entity_id"],
                    "source": "source3",
                    "block_key": block_key
                })

    return pd.DataFrame(pairs)