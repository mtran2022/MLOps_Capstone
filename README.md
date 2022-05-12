This application is designed to predict whether a credit card transaction is fraudulent.

Dataset: Capital One Data Science Challenge https://github.com/CapitalOneRecruiting/DS
Model: XGBoost Classifier https://xgboost.readthedocs.io/en/stable/python/python_intro.html

#### Initial MVP
```
Initial MVP result with initial training+validation dataset of ~500K records, using 6 original features.
Best F1-score was 0.07.
```

After several iterations of feature engineering, 3 additional features were added to the dataset, for a total of 9 features.
Model training with 9 features used 67K records.
Best F1-score was 0.085.

