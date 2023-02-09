# Heart-Disease-Interpretable-Classifier
 An Exploration to find an Interpretable Classifier for Heart Disease dataset with low samples
 
## Introduction
It is important to consider interpretability when building medical ML models. Interpretability refers to the ability to explain or understand the reasoning behind a model's predictions. In the medical field, it is crucial for physicians and other medical professionals to understand the reasoning behind a model's diagnoses or treatment recommendations, as it can have a significant impact on patient care. Additionally, interpretability can help with model validation and identifying potential biases in the data. Overall, interpretability is important in medical ML models as it can improve trust in the model, enhance decision-making and improve patient outcomes.
To practice this concept, I chose a small dataset from the UCI database about [Heart disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
This dataset has multiple files that I have used [processed.cleveland](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data).

To achieve the goal of this exercise, the presence of heart disease in the patient should be to predict. 
The classic machine learning classification methods will be used.

Results of these models have got compared with metrics such as F1 Score and AUC, and analysis of The ROC curves and Recall-Precision curves.


## Data Set Information:
This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to
this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).

### Attributes Information:
    -- 1.age: age in years 
    -- 2.sex: sex (1 = male; 0 = female) 
    -- 3.cp: chest pain type
        ^ Value 1: typical angina
        ^ Value 2: atypical angina
        ^ Value 3: non-anginal pain
        ^ Value 4: asymptomatic 
    -- 4.trestbps: resting blood pressure (in mm Hg on admission to the hospital) 
    -- 5.chol: serum cholestoral in mg/dl 
    -- 6.fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
    -- 7.restecg: resting electrocardiographic results
        ^ Value 0: normal
        ^ Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        ^ Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
    -- 8.thalach: maximum heart rate achieved 
    -- 9.exang: exercise induced angina (1 = yes; 0 = no) 
    -- 10.oldpeak: ST depression induced by exercise relative to rest 
    -- 11.slope: the slope of the peak exercise ST segment
        ^ Value 1: upsloping
        ^ Value 2: flat
        ^ Value 3: downsloping 
    -- 12.ca: number of major vessels (0-3) colored by flourosopy 
    -- 13.thal: 3 = normal; 6 = fixed defect; 7 = reversable defect 
    -- 14.the predicted attribute: diagnosis of heart disease (angiographic disease status)
        ^ Value 0: Absence of heart disease
        ^ Value 1: Presence of heart disease

### Data types:
The dataset columns have loading with specific categorical formats Based on Attributes Information. The written function is in `project_lib/data_loaders.py -> load_data`

# Project's library
The `project_lib` is a package that developed to simplify the process of discovery. It contains tools to train various models, then easily record and compare model's results.

## BinaryClassificationModelsScoring class
This class designed to keep candidates models results in side of capability to test multimodel in parallel discovery.
A sample usage displayed in `autoscoringtable.ipynb` file.

## AnalysisCurvesDisplay
This class give capability to compare at most 10 models with side plots of Target Probability, Calibration Curve, ROC Curve and Precision-Recall Curve.
A sample usage displayed in `AnalysisCurves example.ipynb` file.