#!/usr/bin/env python
# coding: utf-8

# # Huggingface Sagemaker - finetune BERT model
# From https://github.com/huggingface/notebooks/blob/main/sagemaker/01_getting_started_pytorch/sagemaker-notebook.ipynb

# # Development environment
# 

# In[ ]:


get_ipython().system('pip install datasets[s3]==2.17.0')


# In[ ]:


import sagemaker.huggingface


# In[ ]:


import sagemaker
import boto3
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

# use sagemaker execution role which has been configured via IAM with required permissions 
role = "arn:aws:iam::739723034235:role/service-role/SageMaker-ExecutionRole-20240119T141388"

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")


# # Prepare data

# In[ ]:


from datasets import load_dataset
from transformers import AutoTokenizer

# tokenizer used in preprocessing
tokenizer_name = 'distilbert-base-uncased'

# dataset used
dataset_name = 'rotten_tomatoes'

# s3 key prefix for the data
s3_prefix = 'samples/datasets/' + dataset_name


# In[ ]:


# load dataset
dataset = load_dataset(dataset_name)

# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# tokenizer helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

# train, validation and test dataset
train_dataset = dataset['train']
eval_dataset = dataset['validation']
test_dataset = dataset['test']

# select smaller subset for faster training
train_dataset = train_dataset.shuffle().select(range(1000)) 
eval_dataset = eval_dataset.shuffle().select(range(100)) 
test_dataset = test_dataset.shuffle().select(range(100)) 

# tokenize dataset
train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

# set format for pytorch
train_dataset =  train_dataset.rename_column("label", "labels")
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset = eval_dataset.rename_column("label", "labels")
eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


# In[ ]:


# save train_dataset to s3
training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/train'
train_dataset.save_to_disk(training_input_path)

# save validation to s3
validation_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/validation'
eval_dataset.save_to_disk(validation_input_path)


# # Train the model

# In[ ]:


get_ipython().system('pygmentize ./scripts/train_sagemaker.py')


# In[ ]:


from sagemaker.huggingface import HuggingFace

# hyperparameters, which are passed into the training job
hyperparameters={'epochs': 1,
                 'train_batch_size': 32,
                 'learning_rate': 2e-5,
                 'warmup_steps': 0,
                 'model_name':'distilbert-base-uncased'
                 }


# In[ ]:


huggingface_estimator = HuggingFace(entry_point='train_sagemaker.py',
                            source_dir='./scripts',
                            instance_type='ml.p3.2xlarge',
                            instance_count=1,
                            role=role,
                            transformers_version='4.26',
                            pytorch_version='1.13',
                            py_version='py39',
                            hyperparameters = hyperparameters)


# In[ ]:


# starting the train job with our uploaded datasets as input
huggingface_estimator.fit({'train': training_input_path, 'test': validation_input_path})


# # Deploy the endpoint

# In[ ]:


predictor = huggingface_estimator.deploy(1, "ml.g4dn.xlarge")


# In[ ]:


sentiment_input= {"inputs":" great movie"}

predictor.predict(sentiment_input)


# # Test the model

# In[ ]:


def map_labels(label):
    mapping = {'LABEL_0': 0, 'LABEL_1': 1}
    return mapping[label]

sentiment_input= {"inputs": test_dataset["text"]}
test_output = predictor.predict(sentiment_input)
test_predictions = [map_labels(item['label']) for item in test_output]


# In[ ]:


# compute accuracy on test set
from sklearn.metrics import accuracy_score
accuracy_score(test_dataset['label'], test_predictions)


# In[ ]:


# show examples of review and labels
import pandas as pd
df = pd.DataFrame({"Review": test_dataset['text'],
                   "Gold label": test_dataset['label'],
                   "Predicted label": test_predictions})
df.head()


# In[ ]:


# clean up
predictor.delete_model()
predictor.delete_endpoint()


# # What to try next
# - How does the experience using Sagemaker training job compare to running the training in a notebook? Which mode of working do you prefer and why?
# - Watch this workshop on Huggingface and AWS Sagemaker https://huggingface.co/docs/sagemaker/getting-started

# In[ ]:




