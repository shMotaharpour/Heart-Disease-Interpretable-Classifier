import pandas as pd
from pathlib import Path
BASE_DATA_ADDRESS = Path('./data/processed.cleveland.data')
class LoadData:
    
    def __init__(self, file_address=BASE_DATA_ADDRESS):
        binary = pd.CategoricalDtype(categories=[0, 1])
        df = pd.read_csv(
            file_address,
            header=None,
            na_values='?',
            names='age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target'.split(', '),
            dtype={
                'sex':binary,
                'cp':pd.CategoricalDtype(categories=range(1,5)),
                'fbs':binary,
                'restecg':pd.CategoricalDtype(categories=range(3)),
                'exang':binary,
                'slope':pd.CategoricalDtype(categories=range(1,4)),
                'ca':pd.CategoricalDtype(categories=range(4)),
                'thal':pd.CategoricalDtype(categories=[3,6,7]),
                'target':pd.CategoricalDtype(categories=range(2)),
            }
        )

        self._data = df.fillna({'ca':0, 'thal':3, 'target':1})
        
    def get_data(self):
        return self._data.copy()


    def get_data_with_primary_features(self):
        return (
            self._data.get(['sex', 'cp', 'restecg', 'exang', 'ca', 'thal']).copy(),
            self._data.get('target').copy()
        )