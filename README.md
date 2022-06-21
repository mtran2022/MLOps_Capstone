## Project Fraud Vision
<br>
<img src="https://user-images.githubusercontent.com/102324956/168498886-b904b791-c0b0-4a54-bcfa-ea4160e7019a.png" width="600">

This application is designed to predict whether fraudulent credit card transactions. <br>
A use case for this solution would be: use a single model for lower-dollar transactions, and the ensemble of models for higher-dollar transactions. <br>
The cost of operating this solution would be commensurate with the transaction dollar-at-risk.

Running the Application
#### Install packages in WSL2 Ubuntu Root
```
sudo apt update
sudo apt install python3-pip python3-dev
sudo pip3 install --upgrade pip
sudo pip3 install virtualenv 
```
#### Install Docker
```
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```
#### Add Docker user and start service
```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
sudo service docker start
docker run hello-world
```
#### Install git and clone this repo
```
sudo apt install git
git clone https://github.com/mtran2022/MLOps_Capstone.git
```

#### Build the image and run
```
virtualenv cc_fraud
source cc_fraud/bin/activate
cd MLOps_Capstone
docker build -t cc_fraud_serv .
docker run -p 127.0.0.1:8080:8080 cc_fraud_serv
```
#### Use the application
```
Prediction endpoint is on localhost at http://127.0.0.1:8080/predict
Send credit card transaction requests to endpoint and receive fraud prediction.
Use demo.py as reference.
```



Data and Model
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

Model performance is stable with retraining on additional data.
```
| Training Records | Test Records | Average Validation F1-score | Test F1-score |
| ---------------- | ------------ | --------------------------- | ------------- |
| 100K             | 50K          | 0.1117                      | 0.1169        |
| 200K             | 100K         | 0.1148                      | 0.1213        |
| 300K             | 150K         | 0.1184                      | 0.1149        |
| 400K             | 200K         | 0.1161                      | 0.1183        |
| 493K             | 243K         | 0.1186                      | 0.1164        |

Ranking of features most useful in the classification task. <br>
<img src="https://user-images.githubusercontent.com/102324956/169159169-e304933d-baf9-47c9-b3c3-bca41c8a897e.png" width="600">
<br>

The model evaluation process is illustrated below. <br>
<img src="https://user-images.githubusercontent.com/102324956/168513347-71356567-9abd-4caf-a4de-016ebb8cdf07.png" width="600">
<br>

#### 2. Model: Neural Network https://keras.io/



