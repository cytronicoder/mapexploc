"""Unified SHAP interface with fallbacks and reporting utilities."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed
except ImportError as exc:
    logger.warning("Optional dependencies not available: %s", exc)
    shap = None
    plt = None
    Parallel = None
    delayed = None


@dataclass
class Explanation:
    """SHAP explanation containing values, interactions, and expected value."""

    shap_values: np.ndarray
    interaction_values: np.ndarray
    expected_value: float | np.ndarray

    def to_json(self) -> str:
        """Convert explanation to JSON string."""
        expected_val = (
            self.expected_value.tolist()
            if isinstance(self.expected_value, np.ndarray)
            else self.expected_value
        )
        return json.dumps(
            {
                "shap_values": self.shap_values.tolist(),
                "interaction_values": self.interaction_values.tolist(),
                "expected_value": expected_val,
            }
        )


class ShapExplainer:
    """SHAP explainer for Random Forest models matching notebook implementation.
    
    This class provides SHAP explanations specifically designed for Random Forest
    models as used in the notebooks, including comprehensive plotting capabilities.
    """

    def __init__(self, model: Any, output_dir: str = "results/figures/shap_analysis"):
        """Initialize SHAP explainer.
        
        Args:
            model: Trained Random Forest model (pipeline with 'rf' step)
            output_dir: Directory to save SHAP plots
        """
        if shap is None:
            raise ImportError("SHAP is required but not installed. Install with: pip install shap")
        
        self.model = model
        self.output_dir = output_dir
        
        # Extract RF model from pipeline
        if hasattr(model, 'named_steps') and 'rf' in model.named_steps:
            self.rf_model = model.named_steps['rf']
        else:
            self.rf_model = model
            
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.rf_model)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Initialized SHAP TreeExplainer for Random Forest")

    def explain_sample(
        self, 
        X_sample: pd.DataFrame,
        sample_size: int = 200,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for a sample of data.
        
        Args:
            X_sample: Features to explain
            sample_size: Number of samples to use for explanation
            random_state: Random state for sampling
        
        Returns:
            Dictionary containing SHAP values and expected values
        """
        # Sample data if needed
        if len(X_sample) > sample_size:
            X_shap = shap.sample(X_sample, sample_size, random_state=random_state)
        else:
            X_shap = X_sample
            
        logger.info("Computing SHAP values for %d samples", len(X_shap))
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_shap)
        expected_value = self.explainer.expected_value
        
        # Handle multi-class case - convert to list format if needed
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        
        return {
            'shap_values': shap_values,
            'expected_value': expected_value,
            'X_sample': X_shap,
            'classes': getattr(self.rf_model, 'classes_', None)
        }

    def plot_summary(self, explanation: Dict[str, Any], save: bool = True) -> None:
        """Generate and save summary SHAP plot.
        
        Args:
            explanation: SHAP explanation dictionary
            save: Whether to save the plot
        """
        if shap is None or plt is None:
            logger.error("Required packages not available for plotting")
            return
            
        shap_values = explanation['shap_values']
        X_sample = explanation['X_sample']
        classes = explanation.get('classes')
        
        # Summary plot for all classes
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=X_sample.columns,
            plot_type="bar",
            class_names=classes,
            show=False
        )
        
        if save:
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/shap_summary_all_classes.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Saved summary plot to %s", f"{self.output_dir}/shap_summary_all_classes.png")

    def plot_individual_explanations(
        self, 
        explanation: Dict[str, Any], 
        max_samples: int = 10,
        save: bool = True
    ) -> None:
        """Generate individual SHAP explanations (force and waterfall plots).
        
        Args:
            explanation: SHAP explanation dictionary
            max_samples: Maximum number of samples to plot
            save: Whether to save plots
        """
        if shap is None or plt is None:
            logger.error("Required packages not available for plotting")
            return
            
        shap_values = explanation['shap_values']
        expected_value = explanation['expected_value']
        X_sample = explanation['X_sample']
        classes = explanation.get('classes', [])
        
        n_samples = min(len(X_sample), max_samples)
        
        # Create directories for each class
        for cls in classes:
            class_dir = f"{self.output_dir}/{cls.replace(' ', '_').lower()}"
            os.makedirs(class_dir, exist_ok=True)
        
        # Generate plots for each class and sample
        for class_index in range(len(shap_values)):
            if classes and class_index < len(classes):
                label = classes[class_index].replace(" ", "_").lower()
            else:
                label = f"class_{class_index}"
                
            class_dir = f"{self.output_dir}/{label}"
            
            for i in range(n_samples):
                try:
                    # Force plot
                    shap.force_plot(
                        expected_value[class_index] if isinstance(expected_value, np.ndarray) else expected_value,
                        shap_values[class_index][i, :],
                        X_sample.iloc[i, :],
                        matplotlib=True,
                        show=False,
                    )
                    if save:
                        plt.savefig(f"{class_dir}/shap_force_sample{i}.png", bbox_inches="tight", dpi=150)
                        plt.close()

                    # Waterfall plot
                    shap.plots._waterfall.waterfall_legacy(
                        expected_value[class_index] if isinstance(expected_value, np.ndarray) else expected_value,
                        shap_values[class_index][i, :],
                        X_sample.iloc[i, :],
                        show=False,
                    )
                    if save:
                        plt.savefig(f"{class_dir}/shap_waterfall_sample{i}.png", bbox_inches="tight", dpi=150)
                        plt.close()
                        
                except Exception as e:
                    logger.warning("Failed to generate plot for class %s, sample %d: %s", label, i, e)
                    continue
        
        logger.info("Generated individual explanations for %d samples", n_samples)

    def plot_decision_plots(self, explanation: Dict[str, Any], save: bool = True) -> None:
        """Generate decision plots for each class.
        
        Args:
            explanation: SHAP explanation dictionary  
            save: Whether to save plots
        """
        if shap is None or plt is None:
            logger.error("Required packages not available for plotting")
            return
            
        shap_values = explanation['shap_values']
        expected_value = explanation['expected_value']
        X_sample = explanation['X_sample']
        classes = explanation.get('classes', [])
        
        for class_index in range(len(shap_values)):
            if classes and class_index < len(classes):
                label = classes[class_index].replace(" ", "_").lower()
            else:
                label = f"class_{class_index}"
                
            class_dir = f"{self.output_dir}/{label}"
            
            try:
                shap.decision_plot(
                    expected_value[class_index] if isinstance(expected_value, np.ndarray) else expected_value,
                    shap_values[class_index],
                    X_sample,
                    show=False
                )
                
                if save:
                    plt.savefig(f"{class_dir}/shap_decision.png", bbox_inches="tight", dpi=150)
                    plt.close()
                    
            except Exception as e:
                logger.warning("Failed to generate decision plot for class %s: %s", label, e)
                continue
        
        logger.info("Generated decision plots for all classes")

    def plot_interaction_summary(self, explanation: Dict[str, Any], save: bool = True) -> None:
        """Generate interaction summary plot.
        
        Args:
            explanation: SHAP explanation dictionary
            save: Whether to save plot
        """
        if shap is None or plt is None:
            logger.error("Required packages not available for plotting")
            return
            
        X_sample = explanation['X_sample']
        
        try:
            # Compute interaction values
            shap_interact = self.explainer.shap_interaction_values(X_sample)
            
            shap.summary_plot(
                shap_interact,
                X_sample,
                feature_names=X_sample.columns,
                plot_type='dot',
                show=False
            )
            
            if save:
                plt.savefig(f"{self.output_dir}/shap_interaction_summary.png", dpi=150, bbox_inches='tight')
                plt.close()
                logger.info("Saved interaction summary plot")
                
        except Exception as e:
            logger.warning("Failed to generate interaction plot: %s", e)

    def generate_all_plots(
        self, 
        X_data: pd.DataFrame,
        sample_size: int = 200,
        max_individual: int = 10,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Generate all SHAP plots as done in the notebook.
        
        Args:
            X_data: Input features
            sample_size: Number of samples for SHAP analysis
            max_individual: Maximum individual plots to generate
            random_state: Random state for sampling
        
        Returns:
            SHAP explanation dictionary
        """
        # Initialize SHAP if needed
        if shap is not None:
            shap.initjs()
        
        # Generate explanations
        explanation = self.explain_sample(X_data, sample_size, random_state)
        
        # Generate all plots
        self.plot_summary(explanation)
        self.plot_individual_explanations(explanation, max_individual)
        self.plot_decision_plots(explanation)
        self.plot_interaction_summary(explanation)
        
        logger.info("Generated all SHAP plots in %s", self.output_dir)
        
        return explanation


def explain(model: Any, features: np.ndarray) -> np.ndarray:
    """Return SHAP values for features using model if SHAP is available.

    This is a simple compatibility function for the original interface.

    Parameters
    ----------
    model : object
        A trained tree-based model (e.g., RandomForestClassifier).
    features : np.ndarray
        Feature matrix to explain.

    Returns
    -------
    np.ndarray
        SHAP values for the input features.
    """
    if shap is None:
        raise RuntimeError("SHAP is not installed")
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(features)


__all__ = ["ShapExplainer", "Explanation", "explain"]


__all__ = ["ShapExplainer", "Explanation", "explain"]
