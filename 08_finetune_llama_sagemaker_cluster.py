#!/usr/bin/env python
# coding: utf-8

# # Finetune OpenLlama with Huggingface and Sagemaker cluster 
# 
# In this notebook, we will perform instruction tuning OpenLlama using a subset of the Dolly 15k Dataset using Sagemaker.
# 
# This notebook as been put together based a few great examples and blogs. Feel free to visit them to learn more about finetuning. 
# 
# - [Fourthbrain Repository Building with Instruction-Tuned LLMs: a Step-by-Step Guide](https://github.com/FourthBrain/Building-with-Instruction-Tuned-LLMs-A-Step-by-Step-Guide)
# - [Fine-tune LLaMA 2 (7-70B) on Amazon SageMaker](https://www.philschmid.de/sagemaker-llama2-qlora)

# Please ensure your Jupyterlab instance is set to the following: **ml.t3.medium**
# 

# # Development environment

# In[ ]:


get_ipython().system('pip install --upgrade datasets[s3]')


# # Prepare dataset

# Let's load our dataset and get an idea of what we're working with!

# In[ ]:


from datasets import load_dataset

# load training data set
dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# Filter for question answering examples
qa_dataset = dolly_dataset.filter(lambda example: example["category"] == "closed_qa")
qa_dataset = qa_dataset.remove_columns("category")



# In[ ]:


import matplotlib.pyplot as plt

def plot_and_filter_sequence_lengths(dataset_obj, max_length=2200):

    # Initialize a list to store the sequence lengths
    sequence_lengths = []

    # list of indices that are too long
    too_long = []

    # Loop over the dataset and get the lengths of text sequences
    for idx, example in enumerate(dataset_obj):
        sequence_lengths.append(len(example['instruction']) + len(example["context"]) + len(example["response"]))
        if sequence_lengths[idx] > max_length:
          too_long.append(idx)

    # Plot the histogram
    plt.hist(sequence_lengths, bins=30)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Distribution of Text Sequence Lengths')
    plt.show()

    return too_long


# In[ ]:


# filter long tail of very long instances
indexes_to_drop = plot_and_filter_sequence_lengths(qa_dataset, max_length=2000)


# In[ ]:


qa_dataset_reduced = qa_dataset.select(
    i for i in range(len(qa_dataset)) if i not in set(indexes_to_drop)
)


# In[ ]:


qa_dataset_reduced


# In[ ]:


import pprint

# We split the dataset into two where test data is used to evaluate at the end.
train_and_test_dataset = qa_dataset_reduced.train_test_split(test_size=0.1)
train_dataset = train_and_test_dataset["train"]

# Dumping the training data to a local file to be used for training.
train_dataset.to_json("training.jsonl")

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(train_dataset[0])


# In[ ]:


def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt


# In[ ]:


from transformers import AutoTokenizer

model_id = "openlm-research/open_llama_3b_v2" 
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
train_dataset = train_dataset.map(template_dataset, remove_columns=list(train_dataset.features))


# print first sample
print(train_dataset[0]["text"])


# In[ ]:


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
lm_dataset = train_dataset.map(
    lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(train_dataset.features)
).map(
    partial(chunk, chunk_length=2048),
    batched=True,
)

# Print total number of samples
print(f"Total number of samples: {len(lm_dataset)}")


# In[ ]:


lm_dataset


# Connect to AWS S3 bucket and upload the training data.

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


# save train_dataset to s3
training_input_path = f's3://{sess.default_bucket()}/processed/llama/dolly/train'
lm_dataset.save_to_disk(training_input_path)

print("uploaded data to:")
print(f"training dataset to: {training_input_path}")


# Okay, now that we have the Dolly 15k dataset pared down to a more reasonable length and uploaded to S3 - let's set up our model training job.
# 
# Unlike the previous exercise, we will not run training interactively in the notebook but will submit a training run to AWS Sagemaker.

# # Finetune the model

# In[ ]:


from transformers import AutoTokenizer

model_id = "openlm-research/open_llama_3b_v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


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
    instance_type        = 'ml.g5.12xlarge',   # instances type used for the training job
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


# How much did it cost to finetune the model?
# Look up the AWS Sagemaker prices at https://aws.amazon.com/sagemaker/pricing/ and complete the cost calculation below.
# 
# 
# In our example for OpenLlama, the SageMaker training job took `XX seconds`. The instance we used costs `$X` per hour for on-demand usage. As a result, the total cost for finetuning was `$X`.

# # What to do next
# - deploy the mode to endpoint and do inference. Look up this blog and try to follow the instructions to deploy your model. https://www.philschmid.de/sagemaker-falcon-llm
# - compare training in the notebook with the Hugginface trainer and Jumpstart. What are pros and cons of each approach?

# 
