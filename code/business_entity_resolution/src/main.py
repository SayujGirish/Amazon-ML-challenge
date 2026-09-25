from preprocessing import load_data
from normalization import light_normalize_source, advanced_normalize_source
from blocking import create_blocks, generate_candidate_pairs
from features import generate_features
from matching import train_models


# Load data
source1, source2, source3, ground_truth = load_data()


# ---------------------------------------------------------
# LIGHT NORMALIZATION
# ---------------------------------------------------------

print("Light normalization...")

source1 = light_normalize_source(source1)
source2 = light_normalize_source(source2)
source3 = light_normalize_source(source3)


# ---------------------------------------------------------
# BLOCKING
# ---------------------------------------------------------

print("Creating blocks...")

source1, source2, source3 = create_blocks(
    source1,
    source2,
    source3
)


# ---------------------------------------------------------
# CANDIDATE GENERATION
# ---------------------------------------------------------

print("Generating candidate pairs...")

candidate_pairs = generate_candidate_pairs(
    source1,
    source2,
    source3
)

print("Total candidate pairs:", len(candidate_pairs))


# ---------------------------------------------------------
# ADVANCED NORMALIZATION
# ---------------------------------------------------------

print("Advanced normalization...")

source1 = advanced_normalize_source(source1)
source2 = advanced_normalize_source(source2)
source3 = advanced_normalize_source(source3)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

print("Generating features...")

feature_data = generate_features(
    candidate_pairs,
    source1,
    source2,
    source3
)

print("Number of feature rows:", len(feature_data))
print(feature_data.head())


# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------

print("Training models...")

trained_models, results, best_model_name, feature_columns = train_models(
    feature_data,
    ground_truth
)

print("Best model:", best_model_name)

print(candidate_pairs.head())

print(
    "Total candidate pairs:",
    len(candidate_pairs)
)