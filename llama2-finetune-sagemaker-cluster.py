#!/usr/bin/env python
# coding: utf-8

# # Fine-tune LLaMA 2 models with Huggingface and Sagemaker cluster
# 
# 

# # Setup

# In[ ]:


get_ipython().system('pip install --upgrade datasets[s3]')


# # Prepare data

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

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")


# In[ ]:


from datasets import load_dataset

# Load dataset from the hub (only load 10% to speed up training)
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:10%]")

print(f"dataset size: {len(dataset)}")
print(dataset[0])


# In[ ]:


def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt


# In[ ]:


print(format_dolly(dataset[0]))


# In[ ]:


from transformers import AutoTokenizer

model_id = "NousResearch/Llama-2-7b-hf" # not gated
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


# In[ ]:


from random import randint
from itertools import chain
from functools import partial


# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample


# apply prompt template per sample
dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
# print first sample
print(dataset[0]["text"])

# empty list to save remainder from batches to use in next batch
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result


# tokenize and chunk dataset
lm_dataset = dataset.map(
    lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
).map(
    partial(chunk, chunk_length=2048),
    batched=True,
)

# Print total number of samples
print(f"Total number of samples: {len(lm_dataset)}")


# In[ ]:


# save train_dataset to s3
training_input_path = f's3://{sess.default_bucket()}/processed/llama/dolly/train'
lm_dataset.save_to_disk(training_input_path)

print("uploaded data to:")
print(f"training dataset to: {training_input_path}")


# # Train the model

# In[ ]:


import time
from sagemaker.huggingface import HuggingFace
from huggingface_hub import HfFolder

# define Training Job Name 
job_name = f'huggingface-qlora-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

# hyperparameters, which are passed into the training job
hyperparameters ={
  'model_id': model_id,                             # pre-trained model
  'dataset_path': '/opt/ml/input/data/training',    # path where sagemaker will save training dataset
  'epochs': 1,                                      # number of training epochs
  'per_device_train_batch_size': 2,                 # batch size for training
  'lr': 2e-4,                                       # learning rate used during training
  'merge_weights': True,                            # wether to merge LoRA into the model (needs more memory)
}

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point          = 'train.py',      # train script
    source_dir           = 'scripts',         # directory which includes all the files needed for training
    instance_type        = 'ml.p3dn.24xlarge',   # instances type used for the training job
    instance_count       = 1,                 # the number of instances used for training
    base_job_name        = job_name,          # the name of the training job
    role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
    volume_size          = 300,               # the size of the EBS volume in GB
    transformers_version = '4.28',            # the transformers version used in the training job
    pytorch_version      = '2.0',             # the pytorch_version version used in the training job
    py_version           = 'py310',           # the python version used in the training job
    hyperparameters      =  hyperparameters,  # the hyperparameters passed to the training job
    environment          = { "HUGGINGFACE_HUB_CACHE": "/tmp/.cache" }, # set env variable to cache models in /tmp
)


# In[ ]:


# define a data input dictonary with our uploaded s3 uris
data = {'training': training_input_path}

# starting the train job with our uploaded datasets as input
huggingface_estimator.fit(data, wait=True)


# In our example for LLaMA 7B, the SageMaker training job took `XX seconds`, which is about `X hours`. The XXX instance we used costs `$X` per hour for on-demand usage. As a result, the total cost for training our fine-tuned LLaMa 2 model was `$X`.

# # Next Steps 
# - deploy the mode to endpoint and do inference
# - compare training in the notebook with the Hugginface trainer and Jumpstart. What are pros and cons of each approach?

# In[ ]:




