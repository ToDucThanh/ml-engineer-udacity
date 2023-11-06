import json
import boto3

endpoint_Name = "sagemaker-scikit-learn-2023-11-06-02-14-14-585"
runtime = boto3.Session().client("sagemaker-runtime")

def lambda_handler(event, context):
    print('Context: ', context)
    print('EventType: ', type(event))
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_Name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(event)
    )
    
    result = response["Body"].read().decode('utf-8')
    sss = json.loads(result)
    
    return {
        'statusCode': 200,
        'headers' : {'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*'},
        'type-result' : str(type(result)),
        'COntent-Type-In' : str(context),
        'body' : json.dumps(sss)
    }
