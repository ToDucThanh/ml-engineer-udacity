"""
Lambda Function 1: serializeImageData

"""

import json
import boto3
import base64

# A low-level client representing Amazon Simple Storage Service (S3)
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input (You may also check lambda test)
    key = event['s3_key']                               ## TODO: fill in
    bucket = event['s3_bucket']                         ## TODO: fill in
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    s3.download_file(bucket, key, "/tmp/image.png")
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:          # Read binary file
        image_data = base64.b64encode(f.read())      # Base64 encode binary data ('image_data' -> class:bytes)

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    
   
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,      # Bytes data
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


"""
Lambda Function 2: ImageClassification
A lambda function with an additional custom layer (sagemaker)

"""

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-05-04-07-10-29-561" ## TODO: fill in

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT)
    
    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image)
    print(inferences)
    
    # We return the data back to the Step Function    
    event["body"]["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': event["body"]
    }


"""
Lambda Function 3: FilterLowThreshold

"""

import json
import ast

THRESHOLD = 0.7


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event["body"]['inferences'] ## TODO: fill in
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(ast.literal_eval(inferences)) > THRESHOLD     ## TODO: fill in (True, if a value exists above 'THRESHOLD')
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event["body"]           
    }