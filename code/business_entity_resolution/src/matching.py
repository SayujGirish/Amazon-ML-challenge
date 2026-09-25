import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import fbeta_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ---------------------------------------------------------
# CREATE LABELS
# ---------------------------------------------------------

def create_labels(candidate_pairs, ground_truth):

    ground_truth_lookup = {}

    for _, row in ground_truth.iterrows():

        source1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids) or str(matched_ids).strip() == "":
            matched_ids = set()
        else:
            matched_ids = set(
                str(matched_ids).split(",")
            )

        ground_truth_lookup[source1_id] = matched_ids

    labels = []

    for _, pair in candidate_pairs.iterrows():

        source1_id = pair["source1_id"]
        candidate_id = pair["candidate_id"]

        matched_ids = ground_truth_lookup.get(
            source1_id, set()
        )

        if candidate_id in matched_ids:
            labels.append(1)
        else:
            labels.append(0)

    candidate_pairs = candidate_pairs.copy()
    candidate_pairs["label"] = labels

    return candidate_pairs


# ---------------------------------------------------------
# PREPARE TRAINING DATA
# ---------------------------------------------------------

def prepare_training_data(feature_data):

    feature_columns = [
        column
        for column in feature_data.columns
        if column not in [
            "source1_id",
            "candidate_id",
            "source",
            "label"
        ]
    ]

    X = feature_data[feature_columns]
    y = feature_data["label"]

    return X, y, feature_columns


# ---------------------------------------------------------
# CREATE MODELS
# ---------------------------------------------------------

def create_models(y_train):

    positive_count = (y_train == 1).sum()
    negative_count = (y_train == 0).sum()

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1
    )

    models = {

        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "SVM": SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbosity=-1
        ),

        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=42
        )
    }

    return models


# ---------------------------------------------------------
# TRAIN AND COMPARE MODELS
# ---------------------------------------------------------

def train_models(feature_data, ground_truth):

    labeled_pairs = create_labels(
        feature_data[
            [
                "source1_id",
                "candidate_id",
                "source"
            ]
        ],
        ground_truth
    )

    feature_data = feature_data.copy()

    feature_data["label"] = labeled_pairs["label"].values

    X, y, feature_columns = prepare_training_data(
        feature_data
    )

    # Split by Source 1 entity.
    # This prevents candidates belonging to the same
    # Source 1 business from appearing in both train/test.

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_indices, validation_indices = next(
        splitter.split(
            X,
            y,
            groups=feature_data["source1_id"]
        )
    )

    X_train = X.iloc[train_indices]
    X_validation = X.iloc[validation_indices]

    y_train = y.iloc[train_indices]
    y_validation = y.iloc[validation_indices]

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_validation))

    print("\nTraining matches:", y_train.sum())
    print("Training non-matches:", (y_train == 0).sum())

    print("\nValidation matches:", y_validation.sum())
    print("Validation non-matches:", (y_validation == 0).sum())

    models = create_models(y_train)

    results = {}
    trained_models = {}

    print("\nModel Results")
    print("-" * 50)

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        score = fbeta_score(
            y_validation,
            predictions,
            beta=0.5
        )

        results[name] = score
        trained_models[name] = model

        print(
            f"{name}: F0.5 = {score:.4f}"
        )

    print("-" * 50)

    best_model_name = max(
    results,
    key=lambda name: results[name]
)

    print(
        f"\nBest validation model: "
        f"{best_model_name}"
    )

    print(
        f"Validation F0.5: "
        f"{results[best_model_name]:.4f}"
    )

    return (
        trained_models,
        results,
        best_model_name,
        feature_columns
    )