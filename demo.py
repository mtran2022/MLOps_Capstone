import requests
import json

cc_fraud_data=[json.loads(line) for line in open('request.txt', 'r')]
# print(cc_fraud_data)

for trans in cc_fraud_data:
    res = requests.post("http://34.152.8.58:8080/predict", json=trans)
    print(res.json())