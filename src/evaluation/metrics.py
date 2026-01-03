"""
Model evaluation and metrics module
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class FloodModelEvaluator:
    """
    Evaluates flood prediction models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_config = config.get('training', {}).get('metrics', [])
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC-ROC (requires probabilities)
        if y_pred_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                logger.warning("Could not calculate AUC-ROC")
                metrics['auc_roc'] = None
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        logger.info(f"Metrics calculated - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: str = None) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Flood', 'Flood'],
                   yticklabels=['No Flood', 'Flood'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: str = None) -> None:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: str = None) -> None:
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.close()
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true, y_pred,
            target_names=['No Flood', 'Flood'],
            digits=4
        )
        
        logger.info(f"\n{report}")
        
        return report
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_pred_proba: np.ndarray = None,
                      model_name: str = "model",
                      save_dir: str = "evaluation") -> Dict[str, Any]:
        """
        Complete evaluation pipeline
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model
            save_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info(f"Evaluating {model_name}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Generate classification report
        report = self.generate_classification_report(y_true, y_pred)
        
        # Create plots
        os.makedirs(save_dir, exist_ok=True)
        
        # Confusion matrix
        cm_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
        self.plot_confusion_matrix(y_true, y_pred, cm_path)
        
        # ROC curve (if probabilities provided)
        if y_pred_proba is not None:
            roc_path = os.path.join(save_dir, f'{model_name}_roc_curve.png')
            self.plot_roc_curve(y_true, y_pred_proba, roc_path)
            
            pr_path = os.path.join(save_dir, f'{model_name}_precision_recall_curve.png')
            self.plot_precision_recall_curve(y_true, y_pred_proba, pr_path)
        
        return {
            'metrics': metrics,
            'classification_report': report
        }
