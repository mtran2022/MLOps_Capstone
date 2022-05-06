import uvicorn
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from fastapi import FastAPI, HTTPException, status, Request, Form
from pydantic import BaseModel
import json
from pandas.api.types import CategoricalDtype

import logging
from logging.config import dictConfig
from app.log_config import log_config # this is your local file

dictConfig(log_config)
logger = logging.getLogger("my_logger")


app = FastAPI()

class Transaction(BaseModel):
  creditLimit: float
  availableMoney: float
  transactionDateTime: str
  transactionAmount: float
  merchantName: str
  acqCountry: str
  merchantCountryCode: str
  posEntryMode: str
  posConditionCode: str
  merchantCategoryCode: str
  currentExpDate: str
  accountOpenDate: str
  dateOfLastAddressChange: str
  cardCVV: str
  enteredCVV: str
  cardLast4Digits: str
  transactionType: str
  currentBalance: float
  cardPresent: int
  expirationDateKeyInMatch: int

@app.on_event("startup")
def load_clf():
  logger.info("server startup")
  with open('app/cat_var_trans_dict.json', 'r') as f:
    global cat_var_trans_dict, XGB_Classifier
    cat_var_trans_dict =  json.load(f)
    XGB_Classifier= XGBClassifier()#enable_categorical=True,use_label_encoder=False)
    XGB_Classifier.load_model('app/XGB_model.json')

@app.get("/")
async def root():
  logger.info("at root")
  return {'message': 'Hello. This the credit card transaction fraud prediction service'}


@app.post("/predict")
async def predict(request: Transaction):
  logger.info("send prediction")
  # print("start predict fucntion")
  model_input = process_input(request)
  result = predict_fraud(model_input)
  return {'IsFraud': int(result)}
  
def process_input(input):
  logger.info("process input data")
  if input.transactionAmount >0 and input.availableMoney >0:
    transamt_to_avail=input.transactionAmount/input.availableMoney
  elif input.transactionAmount==0:
    transamt_to_avail=0
  elif input.availableMoney <0:
    transamt_to_avail=(abs(input.availableMoney) + input.transactionAmount)/100

  expirationDateKeyInMatch=input.expirationDateKeyInMatch
  merchantCountryCode=input.merchantCountryCode
  merchantCategoryCode=input.merchantCategoryCode
  posEntryMode=input.posEntryMode
  posConditionCode=input.posConditionCode
  cardPresent=input.cardPresent
  is_near_holiday=0
  addr_chg_date_to_trans_date_days=(pd.to_datetime(input.transactionDateTime) \
                                   - pd.to_datetime(input.dateOfLastAddressChange)).days

  columns = ['expirationDateKeyInMatch','merchantCountryCode','merchantCategoryCode'
              ,'posEntryMode','posConditionCode'
              ,'cardPresent','transamt_to_avail','addr_chg_date_to_trans_date_days','is_near_holiday']
  input_df = pd.DataFrame([[expirationDateKeyInMatch
                      ,merchantCountryCode
                      ,merchantCategoryCode
                      ,posEntryMode
                      ,posConditionCode
                      ,cardPresent
                      ,transamt_to_avail                     
                      ,addr_chg_date_to_trans_date_days
                      ,is_near_holiday]] ,columns=columns)

  for col in ['expirationDateKeyInMatch','merchantCountryCode','merchantCategoryCode'
              ,'posEntryMode','posConditionCode','cardPresent','is_near_holiday']:
    sorted_unique_cat_values = cat_var_trans_dict.get(col)          
    cat_type = CategoricalDtype(categories=sorted_unique_cat_values, ordered=False)
    input_df[col]=input_df[col].astype(cat_type)
  
  # for col in ['expirationDateKeyInMatch','merchantCountryCode','merchantCategoryCode'
  #             ,'posEntryMode','posConditionCode'
  #             ,'cardPresent','transamt_to_avail','addr_chg_date_to_trans_date_days','is_near_holiday']:
  #   print('columns',input_df.loc[:,col])
  
  return input_df

def predict_fraud(model_input_df):
  logger.info("make fraud prediction")
  return XGB_Classifier.predict(model_input_df)


# command to run uvicorn
# uvicorn CC_fraud_model:app --port 8000

# command to provide input - IsFraud=1
# curl -X POST -H "Content-Type: application/json" -d '{"creditLimit": 1000.0 ,"availableMoney":71.86 ,"transactionDateTime":"2016-01-26 18:51:45" ,"transactionAmount":237.64 ,"merchantName":"cheapfast.com" ,"acqCountry":"US" ,"merchantCountryCode":"US" ,"posEntryMode":"09" ,"posConditionCode":"01" ,"merchantCategoryCode":"online_retail" ,"currentExpDate":"2026-12-31" ,"accountOpenDate":"2015-11-11" ,"dateOfLastAddressChange":"2015-07-19" ,"cardCVV":"157" ,"enteredCVV":"157" ,"cardLast4Digits":"8924" ,"transactionType":"PURCHASE" ,"currentBalance":928.14 ,"cardPresent":0 ,"expirationDateKeyInMatch":0}' http://localhost:8080/predict
# curl -X POST -H "Content-Type: application/json" -d '{"creditLimit": 2500.0 ,"availableMoney":171.46 ,"transactionDateTime":"2016-02-12 00:38:11" ,"transactionAmount":238.66 ,"merchantName":"Lyft" ,"acqCountry":"US" ,"merchantCountryCode":"US" ,"posEntryMode":"09" ,"posConditionCode":"01" ,"merchantCategoryCode":"rideshare" ,"currentExpDate":"2026-05-31" ,"accountOpenDate":"2013-02-07" ,"dateOfLastAddressChange":"2013-02-07" ,"cardCVV":"153" ,"enteredCVV":"153" ,"cardLast4Digits":"2737" ,"transactionType":"PURCHASE" ,"currentBalance":2328.54 ,"cardPresent":0 ,"expirationDateKeyInMatch":0}' http://localhost:8080/predict

# command to provide input - IsFraud=0
# curl -X POST -H "Content-Type: application/json" -d '{"creditLimit": 15000 ,"availableMoney":4809.54 ,"transactionDateTime":"2016-12-30 22:31:53" ,"transactionAmount":401.49 ,"merchantName":"American Airlines" ,"acqCountry":"US" ,"merchantCountryCode":"US" ,"posEntryMode":"09" ,"posConditionCode":"01" ,"merchantCategoryCode":"airline" ,"currentExpDate":"2021-09-30" ,"accountOpenDate":"2014-08-24" ,"dateOfLastAddressChange":"2016-11-21" ,"cardCVV":"513" ,"enteredCVV":"513" ,"cardLast4Digits":"8971" ,"transactionType":"PURCHASE" ,"currentBalance":10190.46 ,"cardPresent":0 ,"expirationDateKeyInMatch":0}' http://localhost:8080/predict
# curl -X POST -H "Content-Type: application/json" -d '{"creditLimit": 5000 ,"availableMoney":3356.52 ,"transactionDateTime":"2016-02-05 13:31:12" ,"transactionAmount":366.38 ,"merchantName":"walmart.com" ,"acqCountry":"US" ,"merchantCountryCode":"US" ,"posEntryMode":"09" ,"posConditionCode":"01" ,"merchantCategoryCode":"online_retail" ,"currentExpDate":"2025-01-31" ,"accountOpenDate":"2015-04-19" ,"dateOfLastAddressChange":"2016-02-04" ,"cardCVV":"165" ,"enteredCVV":"165" ,"cardLast4Digits":"3078" ,"transactionType":"PURCHASE" ,"currentBalance":1643.48 ,"cardPresent":0 ,"expirationDateKeyInMatch":0}' http://localhost:8080/predict
