import pandas as pd
import re
from xgboost import XGBClassifier
import xgboost as xgb

from keras.models import load_model
from keras.layers import InputLayer
from keras import backend as K

from sklearn.preprocessing import StandardScaler
import joblib

from fastapi import FastAPI, HTTPException, status, Request, Form
from pydantic import BaseModel
import json
from pandas.api.types import CategoricalDtype
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

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
  modelNum: int

@app.on_event("startup")
def load_clf():
  logger.info("server startup")

  global cat_var_trans_dict, XGB_Classifier, top100_retailers_2015_ls ,NNClassifier ,scaler

  XGB_Classifier= XGBClassifier()#enable_categorical=True,use_label_encoder=False)
  XGB_Classifier.load_model('app_2mod/XGB_model.json')

  NNClassifier = load_model('app_2mod/nn_model.h5',custom_objects={'f1':f1})

  with open('app_2mod/cat_var_trans_dict.json', 'r') as f:
    cat_var_trans_dict =  json.load(f)

  with open('app_2mod/top100_retailers_2015.csv', 'r') as f:
    top100_retailers_2015 = pd.read_csv(f ,skipinitialspace=True)
    top100_retailers_2015_ls = [col.lower() for col in top100_retailers_2015.columns]

  scaler = joblib.load('app_2mod/scaler.joblib')

@app.get("/")
async def root():
  logger.info("at root")
  return {'message': 'Hello. This the credit card transaction fraud prediction service'}


@app.post("/predict")
async def predict(request: Transaction):
  logger.info("send prediction")
  # print("start predict fucntion")
  model_input_xgb = process_input_xgb(request)
  result_xgb = predict_fraud_xgb(model_input_xgb)

  model_input_nn = process_input_nn(request)
  result_nn = predict_fraud_nn(model_input_nn)

  if request.modelNum==0:
    response={'Model0': {'IsFraud': int(result_xgb)}}
  elif request.modelNum==1:
    response={'Model1': {'IsFraud': int(result_nn)}}
  elif request.modelNum==2:
    response={'Model0': {'IsFraud': int(result_xgb)} ,'Model1': {'IsFraud': int(result_nn)}}

  return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)



def process_input_xgb(input):
  logger.info("process input data")
 
  expirationDateKeyInMatch=input.expirationDateKeyInMatch
  merchantCountryCode=input.merchantCountryCode
  merchantCategoryCode=input.merchantCategoryCode
  transactionAmount=input.transactionAmount
  posEntryMode=input.posEntryMode
  posConditionCode=input.posConditionCode
  cardPresent=input.cardPresent

  #calculate transamt_to_avail
  if input.transactionAmount >0 and input.availableMoney >0:
    transamt_to_avail=input.transactionAmount/input.availableMoney
  elif input.transactionAmount==0:
    transamt_to_avail=0
  elif input.availableMoney <0:
    transamt_to_avail=(abs(input.availableMoney) + input.transactionAmount)/100


  #transform merchant name to compare to top100_retailers_2015
  merchantName_clean= input.merchantName.replace('.com','')
  if re.match(r'.+#',merchantName_clean):
    merchantName_clean = merchantName_clean[:re.match(r'.+#',merchantName_clean).end()-1].strip().lower()
  else:
    merchantName_clean = merchantName_clean.strip().lower()

  is_top_merchant= int(merchantName_clean in top100_retailers_2015_ls)
  # print('merchantName_clean:',merchantName_clean)
  # print('is_top_merchant:',is_top_merchant)

  columns = XGB_Classifier.get_booster().feature_names

  input_df = pd.DataFrame([[expirationDateKeyInMatch
                            ,merchantCountryCode
                            ,merchantCategoryCode
                            ,transactionAmount
                            ,posEntryMode
                            ,posConditionCode
                            ,cardPresent
                            ,transamt_to_avail                     
                            ,merchantName_clean
                            ,is_top_merchant]] ,columns=columns)

  for col in cat_var_trans_dict.keys():
    sorted_unique_cat_values = cat_var_trans_dict.get(col)          
    cat_type = CategoricalDtype(categories=sorted_unique_cat_values, ordered=False)
    input_df[col]=input_df[col].astype(cat_type)
  
  # for col in ['expirationDateKeyInMatch','merchantCountryCode','merchantCategoryCode'
  #             ,'posEntryMode','posConditionCode'
  #             ,'cardPresent','transamt_to_avail','addr_chg_date_to_trans_date_days','is_near_holiday']:
  #   print('columns',input_df.loc[:,col])
  
  return input_df

def predict_fraud_xgb(model_input_df):
  logger.info("make fraud prediction")
  return XGB_Classifier.predict(model_input_df)


# NN model
def process_input_nn(input):
  logger.info("process input data")
 
  expirationDateKeyInMatch=input.expirationDateKeyInMatch
  merchantCountryCode=input.merchantCountryCode
  merchantCategoryCode=input.merchantCategoryCode
  transactionAmount=input.transactionAmount
  posEntryMode=input.posEntryMode
  posConditionCode=input.posConditionCode
  cardPresent=input.cardPresent

  #calculate transamt_to_avail
  if input.transactionAmount >0 and input.availableMoney >0:
    transamt_to_avail=input.transactionAmount/input.availableMoney
  elif input.transactionAmount==0:
    transamt_to_avail=0
  elif input.availableMoney <0:
    transamt_to_avail=(abs(input.availableMoney) + input.transactionAmount)/100

  #transform merchant name to compare to top100_retailers_2015
  merchantName_clean= input.merchantName.replace('.com','')
  if re.match(r'.+#',merchantName_clean):
    merchantName_clean = merchantName_clean[:re.match(r'.+#',merchantName_clean).end()-1].strip().lower()
  else:
    merchantName_clean = merchantName_clean.strip().lower()

  is_top_merchant= int(merchantName_clean in top100_retailers_2015_ls)
  # print('merchantName_clean:',merchantName_clean)
  # print('is_top_merchant:',is_top_merchant)

  #get column names from model
  columns=[]
  for layer in NNClassifier.layers:
      if isinstance(layer, InputLayer):
          columns.append(layer.name)

  # print('columns:',columns)
  input_df = pd.DataFrame([[expirationDateKeyInMatch,merchantCountryCode,merchantCategoryCode
                            ,posEntryMode,posConditionCode,cardPresent,merchantName_clean
                            ,is_top_merchant,transactionAmount,transamt_to_avail
                            ]] ,columns=columns)
  # print('input_df:',input_df)
  
  #convert categorical variables to category codes
  for col in cat_var_trans_dict.keys():
    sorted_unique_cat_values = cat_var_trans_dict.get(col)          
    cat_type = CategoricalDtype(categories=sorted_unique_cat_values, ordered=False)
    input_df[col]=input_df[col].astype(cat_type).cat.codes
  
  #scale continuous variables
  for col in columns: 
    if col not in cat_var_trans_dict.keys():
      print('cont var:',col)
      input_df[col] = scaler.transform(input_df[col].values.reshape(-1,1))
  
  # print('input_df after transform:\n',input_df)
  
  input_tensor=[]
  for col in columns:
    input_tensor.append(input_df[col].values)
  # print('input_tensor:',input_tensor)
  return input_tensor

def predict_fraud_nn(model_input):
  logger.info("make fraud prediction")
  return NNClassifier.predict(model_input)


def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    
    return f1_val

# command to run uvicorn
# uvicorn CC_fraud_model:app --port 8000

# command to provide input - IsFraud=1
# curl -X POST -H "Content-Type: application/json" -d '{"creditLimit": 5000 ,"availableMoney":1463.88 ,"transactionDateTime":"2016-03-14 14:14:27" ,"transactionAmount":489.11 ,"merchantName":"Fresh eCards" ,"acqCountry":"US" ,"merchantCountryCode":"US" ,"posEntryMode":"02" ,"posConditionCode":"01" ,"merchantCategoryCode":"online_gifts" ,"currentExpDate":"2032-11-30" ,"accountOpenDate":"2014-06-21" ,"dateOfLastAddressChange":"2016-03-11" ,"cardCVV":"869" ,"enteredCVV":"869" ,"cardLast4Digits":"593" ,"transactionType":"PURCHASE" ,"currentBalance":3536.12 ,"cardPresent":0 ,"expirationDateKeyInMatch":0}' http://localhost:8080/predict
# curl -X POST -H "Content-Type: application/json" -d '{"creditLimit": 2500.0 ,"availableMoney":171.46 ,"transactionDateTime":"2016-02-12 00:38:11" ,"transactionAmount":238.66 ,"merchantName":"Lyft" ,"acqCountry":"US" ,"merchantCountryCode":"US" ,"posEntryMode":"09" ,"posConditionCode":"01" ,"merchantCategoryCode":"rideshare" ,"currentExpDate":"2026-05-31" ,"accountOpenDate":"2013-02-07" ,"dateOfLastAddressChange":"2013-02-07" ,"cardCVV":"153" ,"enteredCVV":"153" ,"cardLast4Digits":"2737" ,"transactionType":"PURCHASE" ,"currentBalance":2328.54 ,"cardPresent":0 ,"expirationDateKeyInMatch":0}' http://localhost:8080/predict

# command to provide input - IsFraud=0
# curl -X POST -H "Content-Type: application/json" -d '{"creditLimit": 15000 ,"availableMoney":12560.82 ,"transactionDateTime":"2016-01-05 17:51:35" ,"transactionAmount":238.69 ,"merchantName":"American Airlines" ,"acqCountry":"US" ,"merchantCountryCode":"US" ,"posEntryMode":"09" ,"posConditionCode":"01" ,"merchantCategoryCode":"airline" ,"currentExpDate":"2022-08-31" ,"accountOpenDate":"2015-07-06" ,"dateOfLastAddressChange":"2015-07-06" ,"cardCVV":"325" ,"enteredCVV":"325" ,"cardLast4Digits":"9787" ,"transactionType":"PURCHASE" ,"currentBalance":2439.18 ,"cardPresent":1 ,"expirationDateKeyInMatch":0}' http://localhost:8080/predict
# curl -X POST -H "Content-Type: application/json" -d '{"creditLimit": 10000.0 ,"availableMoney":1780.91 ,"transactionDateTime":"2016-03-18 12:05:40" ,"transactionAmount":168.58 ,"merchantName":"walmart.com" ,"acqCountry":"US" ,"merchantCountryCode":"US" ,"posEntryMode":"09" ,"posConditionCode":"01" ,"merchantCategoryCode":"online_retail" ,"currentExpDate":"2028-02-29" ,"accountOpenDate":"2015-06-19" ,"dateOfLastAddressChange":"2016-01-14" ,"cardCVV":"233" ,"enteredCVV":"233" ,"cardLast4Digits":"4808" ,"transactionType":"PURCHASE" ,"currentBalance":8219.09 ,"cardPresent":0 ,"expirationDateKeyInMatch":0}' http://localhost:8080/predict
# curl -X POST -H "Content-Type: application/json" -d '{"creditLimit": 2500.0 ,"availableMoney":1292.45 ,"transactionDateTime":"2016-01-17 05:23:54" ,"transactionAmount":14.72 ,"merchantName":"AMC #128743" ,"acqCountry":"US" ,"merchantCountryCode":"US" ,"posEntryMode":"05" ,"posConditionCode":"01" ,"merchantCategoryCode":"entertainment" ,"currentExpDate":"2026-08-31" ,"accountOpenDate":"2012-05-17" ,"dateOfLastAddressChange":"2012-05-17" ,"cardCVV":"375" ,"enteredCVV":"375" ,"cardLast4Digits":"3557" ,"transactionType":"PURCHASE" ,"currentBalance":1207.55 ,"cardPresent":1 ,"expirationDateKeyInMatch":0}' http://localhost:8080/predict