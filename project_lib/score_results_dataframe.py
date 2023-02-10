import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from project_lib import LoadData


class BinaryClassificationModelsScoring:
    from sklearn.metrics import (
        accuracy_score, recall_score, precision_score,
        roc_auc_score, roc_auc_score, f1_score, average_precision_score,
    )
    _scoring_funcs = {}
    _instance = {}

    @classmethod
    def tables(cls):
        return cls._instance.keys()

    @property
    def scoring_functions(self):
        return tuple(self._scoring_funcs.keys())

    def __new__(cls, table_name, *args, **kwargs):

        if not len(cls._scoring_funcs):
            # Creates score functions dictionary
            cls._scoring_funcs = {
                a.replace('_score', ''): b
                for a, b in cls.__dict__.items()
                if '_score' in a
            }

        if not all(hasattr(cls, x) for x in
                   'x_train, x_test, y_train, y_test'.split(', ')):
            # Defines same data for all models
            dx, dy = LoadData().get_data_with_primary_features()
            cls.x_train, cls.x_test, cls.y_train, cls.y_test = train_test_split(
                dx, dy, test_size=23, random_state=44)

        return super().__new__(cls)

    def __init__(self, table_name, *, directory='./ScoresTables'):
        self.table_name = table_name
        if table_name in self._instance:
            # Creates results dataframes as singleton
            self._df = self._instance[table_name]
        if not hasattr(self, '_df'):
            self.file_path = Path(f'{directory}/{table_name}.pkl')
            if self.file_path.is_file():
                self._df = self.read_frame()
            else:
                Path(directory).mkdir(
                    parents=True, exist_ok=True)
                self._df = self.new_frame()
            self._instance[table_name] = self._df

            # Stores last model's results
        self._last_result = None

    def read_frame(self):
        return pd.read_pickle(self.file_path)

    def new_frame(self):
        columns = pd.MultiIndex.from_product(
            ['train test'.split(), self._scoring_funcs],
            names=['Level', 'score'])
        df = pd.DataFrame(columns=columns)
        df.index.name = f'{self.table_name} models'
        return df

    @property
    def score_table(self):
        return self._df.copy()

    def __repr__(self):
        return f"[{self.table_name}] scores table with {len(self._df)} records"

    def __del__(self):
        self._df.to_pickle(self.file_path)

    def train_and_scoring(self, model, model_name=''):
        model.fit(self.x_train, self.y_train)
        self._last_result = self.new_frame()
        self._scoring(model, model_name, 'train')
        self._scoring(model, model_name, 'test')
        return self.last_result

    def _scoring(self, model, model_name='', level: 'train or test' = 'train'):
        assert level in 'train test'.split()
        y_true = self.y_test if level == 'test' else self.y_train
        x_in = self.x_test if level == 'test' else self.x_train
        prd = model.predict(x_in)
        prd_p = model.predict_proba(x_in)[:, 1]
        for name, score in self._scoring_funcs.items():
            try:
                s = score(y_true, prd_p)
            except ValueError:
                s = score(y_true, prd)
            finally:
                self._last_result.loc[model_name, (level, name)] = round(s, 3)

    @property
    def last_result(self):
        return self._last_result.copy() if self._last_result is not None else self.new_frame()

    def dump_last_result(self, new_name: str | None = None):
        if self._last_result is not None:
            name = new_name or self._last_result.index[0]
            self._df.loc[name, :] = self._last_result.iloc[0, :]

    def save_table(self):
        self._df.to_pickle(self.file_path)
