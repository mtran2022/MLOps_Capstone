Project Cr

![image](https://user-images.githubusercontent.com/102324956/168498886-b904b791-c0b0-4a54-bcfa-ea4160e7019a.png)

This application is designed to predict whether a credit card transaction is fraudulent.

Dataset: CapOne https://github.com/CapitalOneRecruiting/DS <br>
Model: XGBoost Classifier https://xgboost.readthedocs.io/en/stable/python/python_intro.html

#### Initial MVP
```
Initial MVP result with initial training dataset of ~500K records, using 6 original features.
Average validation F1-score was 0.065.
```
#### Data and Model Iteration
```
Data iterations included feature engineering resulting in 3 additional features added to the dataset, for a total of 9 features.
Initial training with 9 features and 100K records was used to search for optimal hyperparameters.

Using 50 randomly selected sets of hyperparameters from below grid, each set was trained over 10 datasets (each with 20K records).
param_grid={
    'scale_pos_weight':range(10,100,10)
    ,'n_estimators': range(500,1501,100)
    ,'learning_rate': [0.0001, 0.001, 0.01, 0.1]
    ,'max_depth': [6,8,10]
    ,'subsample':[0.5 ,1]
    ,'gamma' :[0.05 ,0.1 ,0.25, 0.5]
    ,'min_child_weight':[100,200,300,400,500,600]
    }
The set of hyperparameters with the best result was:
{'subsample': 0.5, 'scale_pos_weight': 20, 'n_estimators': 800, 'min_child_weight': 100, 'max_depth': 6, 'learning_rate': 0.01, 'gamma': 0.1}
Average validation F1-score was 0.0833.
```
#### Model Retraining
```
Retraining the model with additional data generally improved the Average Validation F1-score.
Comparison between the Average Validation F1-score and Test F1-score shows the model generalizes well to unseen data.

Model performance begins to decline with 400K training records.
This may signal a change in the data-generating process that requires development of a new model.

| Training Records | Test Records | Average Validation F1-score | Test F1-score |
| ---------------- | ------------ | --------------------------- | ------------- |
| 100K             | 50K          | 0.0833                      | 0.1025        |
| 200K             | 100K         | 0.0887                      | 0.0864        |
| 300K             | 150K         | 0.0922                      | 0.0914        |
| 400K             | 200K         | 0.0867                      | 0.0922        |
| 493K             | 243K         | 0.0851                      | 0.0904        |

```
The model evaluation process is illustrated below.
![image](https://user-images.githubusercontent.com/102324956/168297824-2a035a69-2919-475c-be12-d657dbff14cd.png)
