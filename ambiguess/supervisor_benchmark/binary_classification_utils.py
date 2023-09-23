"""Provides a neat way to calculate AUC-ROC for various outlier sets."""
import numpy as np
from sklearn.metrics import roc_auc_score


class Evaluation:
    """Essentially an AUC-ROC calculator, keeping nominal uncertainties as state."""

    def __init__(self, nominal_uncertainties: np.ndarray):
        self.nominal_uncertainties = nominal_uncertainties

    # docstr-coverage:excused `no one is reading this anyways`
    @staticmethod
    def __is_misclassified(predictions: np.ndarray, y_true: np.ndarray):
        return predictions != y_true

    def auc_roc(self, uncertainties: np.ndarray):
        """Calculate the area under the precision-recall curve."""

        truth = np.concatenate((np.zeros(len(self.nominal_uncertainties)), np.ones(len(uncertainties))))
        uncertainties = np.concatenate((self.nominal_uncertainties, uncertainties))

        uncertainties[uncertainties == np.inf] = 1e10
        uncertainties[uncertainties == -np.inf] = -1e10

        return roc_auc_score(truth, uncertainties)
