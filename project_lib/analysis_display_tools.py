import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay

class AnalysisCurvesDisplay:
    
    @staticmethod
    def create_frames():
        fig, ((axlu, axru), (axld, axrd)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(13, 11))
        axlu.set_title('Target Probability')
        axlu.set_xlabel('Predicted probability')
        axld.set_title('ROC Curve')
        axrd.set_title('Precision Recall Curve')
        axru.set_title('Calibration Curve')
        return fig, (axlu, axru, axld, axrd)
        
        
    def __init__(self,data:'pd.DataFrame', name='Classifier'):
        if not {'prob', 'target'} <= set(data.columns):
            raise ValueError("'prob' or 'target' columns does not exist")
            
        self.name = name
        self.data = data
        pos_mask = data.target == 1
        self.pos_data = data.loc[pos_mask]
        self.neg_data = data.loc[~pos_mask]
            
    def plot(self, ax_probability, ax_calib, ax_roc, ax_per_recall, **kwargs):
        ax_probability.boxplot(
            x=[self.pos_data.prob, self.neg_data.prob],
            vert=False,
            labels=['Positive targets', 'Negative targets']
        )
        
        args = (self.data.target, self.data.prob)
        rcp = RocCurveDisplay.from_predictions(*args, ax=ax_roc, name=self.name, **kwargs)
        prp = PrecisionRecallDisplay.from_predictions(*args, ax=ax_per_recall, name=self.name, **kwargs)
        calp =  CalibrationDisplay.from_predictions(*args, strategy='quantile', n_bins=23, ax=ax_calib, name=self.name, **kwargs)
        
    