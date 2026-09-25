from preprocessing import load_data

source1, source2, sourcafrom preprocessing import load_data
from normalization import normalize_all_sources
from blocking import create_blocks, generate_candidate_pairs


source1, source2, source3, ground_truth = load_data()


# Normalization
source1, source2, source3 = normalize_all_sources(
    source1,
    source2,
    source3
)


# Blocking
source1, source2, source3 = create_blocks(
    source1,
    source2,
    source3
)


# Generate candidate pairs
candidate_pairs = generate_candidate_pairs(
    source1,
    source2,
    source3
)


print(candidate_pairs.head())
print("Total candidate pairs:", len(candidate_pairs))e3, ground_truth = load_data()