import pandas as pd


def create_block_keys(row):
    country = row["country_normalized"]
    name = row["business_name_normalized"]
    address = row["business_address_normalized"]

    keys = set()

    # Block 1: country + first 3 characters of business name
    if country and name:
        keys.add(f"name3|{country}|{name[:3]}")

    # Block 2: country + first 3 characters of address
    if country and address:
        keys.add(f"address3|{country}|{address[:3]}")

    return keys


def create_blocks(source1, source2, source3):
    source1 = source1.copy()
    source2 = source2.copy()
    source3 = source3.copy()

    source1["block_keys"] = source1.apply(create_block_keys, axis=1)
    source2["block_keys"] = source2.apply(create_block_keys, axis=1)
    source3["block_keys"] = source3.apply(create_block_keys, axis=1)

    return source1, source2, source3


def build_block_index(source):
    block_index = {}

    for _, row in source.iterrows():
        entity_id = row["entity_id"]

        for block_key in row["block_keys"]:
            if block_key not in block_index:
                block_index[block_key] = []

            block_index[block_key].append(entity_id)

    return block_index


def generate_candidate_pairs(source1, source2, source3):
    source2_index = build_block_index(source2)
    source3_index = build_block_index(source3)

    candidate_pairs = set()

    for _, s1 in source1.iterrows():
        s1_id = s1["entity_id"]

        for block_key in s1["block_keys"]:

            # Source 2 candidates
            for s2_id in source2_index.get(block_key, []):
                candidate_pairs.add(
                    (s1_id, s2_id, "source2")
                )

            # Source 3 candidates
            for s3_id in source3_index.get(block_key, []):
                candidate_pairs.add(
                    (s1_id, s3_id, "source3")
                )

    return pd.DataFrame(
        list(candidate_pairs),
        columns=[
            "source1_id",
            "candidate_id",
            "source"
        ]
    )