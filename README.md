## Project Fraud Vision
<br>
<img src="https://user-images.githubusercontent.com/102324956/168498886-b904b791-c0b0-4a54-bcfa-ea4160e7019a.png" width="600">

This application is designed to predict whether a credit card transaction is fraudulent.

Dataset: CapOne https://github.com/CapitalOneRecruiting/DS <br>

#### 1. Model: XGBoost Classifier https://xgboost.readthedocs.io/en/stable/python/python_intro.html
#### Initial MVP
```
Initial MVP result with training dataset of ~500K records, using 7 original features identified as differentiating.
Average validation F1-score was 0.065.
```
#### Data and Model Iteration
```
Data iterations included feature engineering resulting in 3 additional features added to the dataset, for a total of 10 features.
Initial training with 10 features and 100K records was used to search for optimal hyperparameters.

Using 50 randomly selected sets of hyperparameters from below grid, each set was trained over 10 datasets (each with 20K records).
param_grid={
    'scale_pos_weight':range(10,50,10)
    ,'n_estimators': range(200,600,100)
    ,'learning_rate': [0.0001, 0.001, 0.01, 0.1]
    ,'max_depth': [4,6,8]
    ,'subsample':[0.5 ,1]
    ,'gamma' :[0.05 ,0.1 ,0.25, 0.5]
    ,'min_child_weight':[100,200,300,400,500,600]
    }
The set of hyperparameters with the best result was:
{'subsample': 1, 'scale_pos_weight': 20, 'n_estimators': 500, 'min_child_weight': 300, 'max_depth': 6, 'learning_rate': 0.1, 'gamma': 0.5}
Average validation F1-score was 0.1117.
```
#### Model Retraining
```
Retraining the model with additional data generally improved the Average Validation F1-score.
Comparison between the Average Validation F1-score and Test F1-score shows the model generalizes well to unseen data.

```
| Training Records | Test Records | Average Validation F1-score | Test F1-score |
| ---------------- | ------------ | --------------------------- | ------------- |
| 100K             | 50K          | 0.1117                      | 0.1169        |
| 200K             | 100K         | 0.1148                      | 0.1213        |
| 300K             | 150K         | 0.1184                      | 0.1149        |
| 400K             | 200K         | 0.1161                      | 0.1183        |
| 493K             | 243K         | 0.1186                      | 0.1164        |

Ranking of features most use in the classification task. <br>
<img src="https://user-images.githubusercontent.com/102324956/169158595-0ed2f140-4340-4bfd-909e-028cd9f88efc.png" width="600">
<br>

The model evaluation process is illustrated below. <br>
<img src="https://user-images.githubusercontent.com/102324956/168513347-71356567-9abd-4caf-a4de-016ebb8cdf07.png" width="600">
<br>

#### 2. Model: Deep Feed-Forward Neural Network
#### Data and Model Iteration
```
Data iterations included feature engineering 
The dataset is normalized before passing it to the model, transformed into a PyTorch dataframe and subsequently a data loader is created to yield batches of it in a random fashion. Initial training with 9 features and 100K records was used to search for optimal hyperparameters.

Using 6 randomly selected sets of hyperparameters. Each each set was trained on a number of units:
- conv_input_size: (38,), input_size: 38, D: 38, output_size: 1
- num_units [38, 50, 50, 10, 25, 25]
```
|    Epoch    | Train Loss  | Test Loss |   Acc   |   AUC |   F1  |
| ----------- | ----------- | --------- |  ------ | ----- | ----- |
| Epoch 01/10 |   0.7490    |  0.7228   |  0.332  | 0.620 | 0.036 |
| Epoch 02/10 |   0.7490    |  0.7216   |  0.426  | 0.624 | 0.038 |
| Epoch 03/10 |   0.7490    |  0.7254   |  0.437  | 0.625 | 0.038 |
| Epoch 04/10 |   0.7487    |  0.7273   |  0.439  | 0.625 | 0.038 |
| Epoch 05/10 |   0.7488    |  0.7279   |  0.438  | 0.625 | 0.038 |
| Epoch 06/10 |   0.7488    |  0.7279   |  0.439  | 0.625 | 0.038 |
| Epoch 07/10 |   0.7484    |  0.7278   |  0.439  | 0.625 | 0.038 |
| Epoch 08/10 |   0.7488    |  0.7280   |  0.439  | 0.625 | 0.038 |
| Epoch 09/10 |   0.7494    |  0.7278   |  0.439  | 0.625 | 0.038 |
| Epoch 10/10 |   0.7487    |  0.7275   |  0.439  | 0.625 | 0.038 |

#### Model Retraining
```
Retraining the model with additional data generally improved the Average Validation F1-score.
Comparison between the Average Validation F1-score and Test F1-score shows the model generalizes well to unseen data.

Train loss: 0.552. Acc: 72.28%. AUC: 0.723. F1: 0.467
Test  loss: 0.480. Acc: 79.50%. AUC: 0.728. F1: 0.071

Confusion matrix, without normalisation
         [122286  30655]
        [  1184   1209]
```

#### High Level Roadmap for next 5 weeks
Develop the remaining 2 models.
Build the ML pipeline with Google Vertex AI.
