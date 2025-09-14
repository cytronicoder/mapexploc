"""Random Forest model utilities."""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv: int = 3,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    use_smote: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Random Forest classifier with hyperparameter tuning and class balancing.

    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for randomized search
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        use_smote: Whether to use SMOTE for class balancing
        random_state: Random state for reproducibility

    Returns:
        Dictionary containing trained model, best parameters, and evaluation results
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.error(
            "Required packages not installed. Install with: pip install scikit-learn imbalanced-learn"
        )
        raise ImportError("scikit-learn and imbalanced-learn are required")

    if param_grid is None:
        param_grid = {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [None, 20],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [1, 2],
        }

    # Check if dataset is too small for SMOTE
    min_class_size = y_train.value_counts().min()
    if use_smote and min_class_size < 6:  # SMOTE needs k=5 neighbors by default
        logger.warning(f"Disabling SMOTE due to small class size ({min_class_size} samples)")
        use_smote = False
    
    # Create pipeline with optional SMOTE and scaling
    steps = []
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.extend(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )

    pipeline = ImbPipeline(steps)

    # Randomized search with stratified cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    logger.info(
        "Starting RandomForest randomized search with %d iterations", 20
    )  # default n_iter for RandomizedSearchCV

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    logger.info("Best Random Forest parameters: %s", search.best_params_)
    logger.info("Best CV %s score: %.4f", scoring, search.best_score_)

    # Convert results to list of dictionaries
    cv_results = []
    for i in range(len(search.cv_results_["mean_test_score"])):
        result = {}
        for key, values in search.cv_results_.items():
            result[key] = values[i]
        cv_results.append(result)

    # Return dictionary with all results
    return {
        'model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'cv_results': cv_results,
        'search': search
    }


def rf_predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict using trained Random Forest model.

    Args:
        model: Trained Random Forest model (pipeline)
        X: Features to predict on

    Returns:
        Predicted labels
    """
    logger.debug("Running Random Forest inference on %d samples", len(X))
    return model.predict(X)


def rf_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict class probabilities using trained Random Forest model.

    Args:
        model: Trained Random Forest model (pipeline)
        X: Features to predict on

    Returns:
        Predicted class probabilities
    """
    logger.debug("Running Random Forest probability inference on %d samples", len(X))
    return model.predict_proba(X)


def evaluate_rf(
    model: Any, X_val: pd.DataFrame, y_val: pd.Series, output_dir: str = "results"
) -> Dict[str, Any]:
    """Comprehensive Random Forest model evaluation.

    Args:
        model: Trained Random Forest model
        X_val: Validation features
        y_val: Validation labels
        output_dir: Directory to save results

    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            classification_report,
            confusion_matrix,
            roc_curve,
            auc,
            precision_recall_curve,
            average_precision_score,
            brier_score_loss,
            calibration_curve,
        )
        from sklearn.preprocessing import label_binarize
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("Required packages not installed")
        raise ImportError(
            "scikit-learn, matplotlib, and seaborn are required for evaluation"
        )

    import os

    # Make predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    # Basic metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    logger.info("Validation accuracy: %.4f", accuracy)
    logger.info("Validation F1-weighted: %.4f", f1_weighted)

    # Get classes
    classes = model.named_steps["rf"].classes_

    # Classification report
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=classes)

    # ROC analysis
    y_val_bin = label_binarize(y_val, classes=classes)

    roc_data = {}
    pr_data = {}
    brier_scores = {}

    for i, cls in enumerate(classes):
        if len(classes) > 2:  # Multi-class
            fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_proba[:, i])
            precision, recall, _ = precision_recall_curve(
                y_val_bin[:, i], y_proba[:, i]
            )
            brier_score = brier_score_loss(y_val_bin[:, i], y_proba[:, i])
        else:  # Binary
            fpr, tpr, _ = roc_curve(y_val_bin, y_proba[:, 1])
            precision, recall, _ = precision_recall_curve(y_val_bin, y_proba[:, 1])
            brier_score = brier_score_loss(y_val_bin, y_proba[:, 1])

        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(
            y_val_bin[:, i] if len(classes) > 2 else y_val_bin, y_proba[:, i]
        )

        roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
        pr_data[cls] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "avg_precision": avg_precision,
        }
        brier_scores[cls] = brier_score

    # Feature importance
    feature_importance = None
    if hasattr(model.named_steps["rf"], "feature_importances_"):
        feature_importance = model.named_steps["rf"].feature_importances_

    # Save results
    if output_dir:
        os.makedirs(f"{output_dir}/csv", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

        # Save metrics
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{output_dir}/csv/rf_classification_report.csv")

        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.to_csv(f"{output_dir}/csv/rf_confusion_matrix.csv")

        # Save feature importance
        if feature_importance is not None:
            feat_df = pd.DataFrame(
                {
                    "feature": range(len(feature_importance)),
                    "importance": feature_importance,
                }
            ).sort_values("importance", ascending=False)
            feat_df.to_csv(f"{output_dir}/csv/rf_feature_importances.csv", index=False)

        # Save ROC and PR data
        roc_auc_df = pd.DataFrame(
            [(cls, data["auc"]) for cls, data in roc_data.items()],
            columns=["class", "auc"],
        )
        roc_auc_df.to_csv(f"{output_dir}/csv/rf_roc_auc_values.csv", index=False)

        pr_avg_df = pd.DataFrame(
            [(cls, data["avg_precision"]) for cls, data in pr_data.items()],
            columns=["class", "avg_precision"],
        )
        pr_avg_df.to_csv(f"{output_dir}/csv/rf_pr_avg_precision.csv", index=False)

        brier_df = pd.DataFrame.from_dict(
            brier_scores, orient="index", columns=["brier_score"]
        )
        brier_df.to_csv(f"{output_dir}/csv/rf_brier_scores.csv")

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes.tolist(),
        "roc_data": roc_data,
        "pr_data": pr_data,
        "brier_scores": brier_scores,
        "feature_importance": (
            feature_importance.tolist() if feature_importance is not None else None
        ),
    }


__all__ = ["train_random_forest", "rf_predict", "rf_predict_proba", "evaluate_rf"]
