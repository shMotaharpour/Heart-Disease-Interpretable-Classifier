from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay

class AnalysisCurvesDisplay:
    
    @staticmethod
    def create_frames():
        fig, ((axlu, axru), (axld, axrd)) = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=False, figsize=(13, 11))
        axlu.set_title('Target Probability')
        axlu.set_title('Target Probability')
        axlu.set_xlabel('Predicted probability')
        axlu.set_ylabel('Actual label')
        axlu.set_yticks(ticks=[0,1], labels=['False', 'True'])
        axlu.set_ylim(-0.5, 2.1)

        axld.set_title('ROC Curve')
        axrd.set_title('Precision Recall Curve')
        axru.set_title('Calibration Curve')
        return fig, (axlu, axru, axld, axrd)
        
        
    def __init__(self,y_true:Iterable, prob_predict:Iterable, name='Classifier'):
        y_true = np.array(y_true)
        if set(y_true) != {0, 1}:
            raise ValueError("`y_true` should be the only content of 1 and 0")
            
        prob_predict = np.array(prob_predict)
        if np.any(prob_predict < 0) or np.any(prob_predict > 1):
            raise ValueError("`prob_predict` should only contain numbers between 0 and 1")
            
        self.name = name
        self.y_true = y_true
        self.prob_predict = prob_predict
        self.args = (y_true, prob_predict)
            
    def plot(self, ax_probability, ax_calib, ax_roc, ax_per_recall,
             **kwargs):

        self._plot_probability(ax_probability, **kwargs)
        rcp = RocCurveDisplay.from_predictions(
            *self.args, ax=ax_roc, name=self.name, **kwargs
        )
        prp = PrecisionRecallDisplay.from_predictions(
            *self.args, ax=ax_per_recall, name=self.name, **kwargs
        )
        calp =  CalibrationDisplay.from_predictions(
            *self.args, strategy='quantile', n_bins=11, ax=ax_calib,
            name=self.name, **kwargs
        )
        
    def _plot_probability(self, ax, extend_factor=0.1, **kwargs):
        leg = ax.get_legend()
        n = len(leg.get_texts()) if leg else 0
        assert n < 9, 'There are too many plots'
        ax.scatter(
            x=self.prob_predict,
            y=self.y_true + 0.1*n,
            alpha=0.1,
            label=self.name,
            **kwargs
        )
        _, y1 = ax.get_ylim()
        ax.set_ybound(None, y1 + extend_factor)
        ax.legend()
        
    