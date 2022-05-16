![image](https://user-images.githubusercontent.com/102324956/168498886-b904b791-c0b0-4a54-bcfa-ea4160e7019a.png)

Project Cr

This application is designed to predict whether a credit card transaction is fraudulent.

Dataset: CapOne https://github.com/CapitalOneRecruiting/DS <br>

#### Initial MVP
```
Initial MVP result with initial training dataset of ~500K records, using 6 original features.
Average validation F1-score was 0.065.

**Models:** 

1. XGBoost Classifier https://xgboost.readthedocs.io/en/stable/python/python_intro.html

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
![image](https://user-images.githubusercontent.com/102324956/16829782 - 4-2a035a69-2919-475c-be12-d657dbff14cd.png)

2. Deep Feed-Forward Neural Network

```
#### Data and Model Iteration
```
Data iterations included feature engineering 
The dataset is normalized before passing it to the model, transformed into a PyTorch dataframe and subsequently a data loader is created to yield batches of it in a random fashion. Initial training with 9 features and 100K records was used to search for optimal hyperparameters.

Using 6 randomly selected sets of hyperparameters. Each each set was trained on a number of units:
- conv_input_size: (38,), input_size: 38, D: 38, output_size: 1
- num_units [38, 50, 50, 10, 25, 25]
Net(
  (layers): ModuleList(
    (0): Linear(in_features=38, out_features=50, bias=False)
    (1): Tanh()
    (2): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=50, out_features=50, bias=False)
    (5): Tanh()
    (6): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.3, inplace=False)
    (8): Linear(in_features=50, out_features=10, bias=False)
    (9): Tanh()
    (10): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): Dropout(p=0.3, inplace=False)
    (12): Linear(in_features=10, out_features=25, bias=False)
    (13): Tanh()
    (14): BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (15): Dropout(p=0.3, inplace=False)
    (16): Linear(in_features=25, out_features=25, bias=False)
    (17): Tanh()
    (18): BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): Dropout(p=0.3, inplace=False)
    (20): Linear(in_features=25, out_features=1, bias=True)
  )
)

Epoch 01/10, Train Loss: 0.7490, Test Loss: 0.7228, Acc: 0.332, AUC: 0.620, F1: 0.036
Epoch 02/10, Train Loss: 0.7490, Test Loss: 0.7216, Acc: 0.426, AUC: 0.624, F1: 0.038
Epoch 03/10, Train Loss: 0.7490, Test Loss: 0.7254, Acc: 0.437, AUC: 0.625, F1: 0.038
Epoch 04/10, Train Loss: 0.7487, Test Loss: 0.7273, Acc: 0.439, AUC: 0.625, F1: 0.038
Epoch 05/10, Train Loss: 0.7488, Test Loss: 0.7279, Acc: 0.438, AUC: 0.625, F1: 0.038
Epoch 06/10, Train Loss: 0.7488, Test Loss: 0.7279, Acc: 0.439, AUC: 0.625, F1: 0.038
Epoch 07/10, Train Loss: 0.7484, Test Loss: 0.7278, Acc: 0.439, AUC: 0.625, F1: 0.038
Epoch 08/10, Train Loss: 0.7488, Test Loss: 0.7280, Acc: 0.439, AUC: 0.625, F1: 0.038
Epoch 09/10, Train Loss: 0.7494, Test Loss: 0.7278, Acc: 0.439, AUC: 0.625, F1: 0.038
Epoch 10/10, Train Loss: 0.7487, Test Loss: 0.7275, Acc: 0.439, AUC: 0.625, F1: 0.038

#### Model Retraining
Retraining the model with additional data generally improved the Average Validation F1-score.
Comparison between the Average Validation F1-score and Test F1-score shows the model generalizes well to unseen data.

Train loss: 0.552. Acc: 72.28%. AUC: 0.723. F1: 0.467
Test  loss: 0.480. Acc: 79.50%. AUC: 0.728. F1: 0.071

Confusion matrix, without normalisation
         [122286  30655]
        [  1184   1209]


