#!/usr/bin/env python
# coding: utf-8

# # Try out Amazon Bedrock
# In this short tutorial you prompt an LLM model on AWS Bedrock.
# 

# In[ ]:


get_ipython().system(' pip install boto3=="1.29.6"')


# In[ ]:


import boto3
import json 
import botocore

bedrock = boto3.client(service_name="bedrock", region_name="us-east-1")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")


# In[ ]:


bedrock.list_foundation_models()


# In[ ]:


# If you'd like to try your own prompt, edit this parameter!
prompt_data = """Command: Write me a poem about machine learning in the style of Shakespeare.

Poem:
"""


# In[ ]:


try:
    
    body = json.dumps({"inputText": prompt_data})
    modelId = "amazon.titan-text-express-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body.get("results")[0].get("outputText"))

except botocore.exceptions.ClientError as error:
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
    else:
        raise error


# In[ ]:




