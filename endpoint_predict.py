from google.cloud import aiplatform

project="mlops-3"
location="northamerica-northeast1"
endpoint="8511416267637260288"
instances=[5000 ,1463.88 ,"2016-03-14 14:14:27" ,489.11 ,"Fresh eCards" ,"US" ,"US" ,"02" ,"01" ,"online_gifts" ,"2032-11-30" ,"2014-06-21" ,"2016-03-11" ,"869" ,"869" ,"593" ,"PURCHASE" ,3536.12 ,0 ,0]

#  [START aiplatform_sdk_endpoint_predict_sample]
def endpoint_predict_sample(
    project: str, location: str, endpoint: str, instances: list):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    print(prediction)
    return prediction

print(endpoint_predict_sample(project,location,endpoint,instances))
