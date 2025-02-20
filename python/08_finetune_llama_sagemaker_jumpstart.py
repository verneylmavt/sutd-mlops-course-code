#!/usr/bin/env python
# coding: utf-8

# # Fine-tune Llama 2 models on SageMaker JumpStart
# In this notebook, we will perform instruction tuning Llama 2 using a subset of the Dolly 15k Dataset using Sagemaker Jumpstart.
# 
# This notebook as been put together based on an example from the Generative AI on AWS book https://github.com/generative-ai-on-aws/generative-ai-on-aws

# # Setup 

# In[ ]:


get_ipython().system(' pip install datasets[s3]')
get_ipython().system(' pip install sagemaker=="2.198.0"')


# In[ ]:


import sagemaker

sagemaker.__version__


# ## Deploy Pre-trained Model
# 
# ---
# 
# First we will deploy the Llama-2 7B model as a SageMaker endpoint and do some inference.
# 
# ---

# In[ ]:


model_id, model_version = "meta-textgeneration-llama-2-7b", "3.1.0"


# In[ ]:


from sagemaker.jumpstart.model import JumpStartModel

pretrained_model = JumpStartModel(model_id=model_id, 
                                  model_version=model_version)
pretrained_predictor = pretrained_model.deploy(accept_eula=True)


# In[ ]:


def print_response(payload, response):
    print(payload["inputs"])
    print(f"> {response[0]['generated_text']}")
    print("\n==================================\n")


# In[ ]:


payload = {
    "inputs": "I believe the meaning of life is",
    "parameters": {
        "max_new_tokens": 64,
        "top_p": 0.9,
        "temperature": 0.6,
        "return_full_text": False,
    },
}
try:
    response = pretrained_predictor.predict(payload, custom_attributes="accept_eula=true")
    print_response(payload, response)
except Exception as e:
    print(e)


# ## Dataset preparation for fine-tuning
# 
# ---
# 
# You can fine-tune on the dataset with domain adaptation format or instruction tuning format. We will use a subset of [Dolly dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) in an instruction tuning format. Dolly dataset contains roughly 15,000 instruction following records for various categories such as question answering, summarization, information extraction etc. It is available under Apache 2.0 license. 
# 
# ---

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
indexes_to_drop = plot_and_filter_sequence_lengths(qa_dataset, max_length=1000)


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


# ---
# Next, we create a prompt template for using the data in an instruction / input format for the training job.

# In[ ]:


import json

template = {
    "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n",
    "completion": "{response}",
}
with open("template.json", "w") as f:
    json.dump(template, f)


# ### Upload dataset to S3
# ---
# 
# We will upload the prepared dataset to S3 which will be used for fine-tuning.
# 
# ---

# In[ ]:


from sagemaker.s3 import S3Uploader
import sagemaker
import random

bucket = sagemaker.Session().default_bucket()
local_data_file = "training.jsonl"
train_data_location = f"s3://{bucket}/dolly_dataset"
S3Uploader.upload(local_data_file, train_data_location)
S3Uploader.upload("template.json", train_data_location)
print(f"Training data: {train_data_location}")


# # Train the model
# 
# Next, we fine-tune the LLaMA v2 7B model. Finetuning scripts are based on scripts provided by [this repo](https://github.com/facebookresearch/llama-recipes/tree/main).
# 

# In[ ]:


from sagemaker.jumpstart.estimator import JumpStartEstimator

estimator = JumpStartEstimator(
    model_id=model_id,
    model_version=model_version,
    instance_type="ml.g5.12xlarge",
    instance_count=1,
    environment={"accept_eula": "true"}
)

# By default, instruction tuning is set to false
estimator.set_hyperparameters(instruction_tuned="True", 
                              epoch="1", 
                              max_input_length="1024")
estimator.fit({"training": train_data_location})


# ### Deploy the fine-tuned model
# ---
# Next, we deploy fine-tuned model. We will compare the performance of fine-tuned and pre-trained model.
# 
# ---

# In[ ]:


finetuned_predictor = estimator.deploy()


# ### Evaluate the pre-trained and fine-tuned model
# ---
# Next, we use the test data to evaluate the performance of the fine-tuned model and compare it with the pre-trained model. 
# 
# ---

# In[ ]:


import pandas as pd
from IPython.display import display, HTML

test_dataset = train_and_test_dataset["test"]

inputs, ground_truth_responses, responses_before_finetuning, responses_after_finetuning = (
    [],
    [],
    [],
    [],
)


def predict_and_print(datapoint):
    # For instruction fine-tuning, we insert a special key between input and output

    input_output_demarkation_key = "\n\n### Response:\n"

    prompt = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{datapoint["instruction"]}\n\n### Input:\n{datapoint["context"]}\n\n'    
    payload = {
        "inputs": prompt + input_output_demarkation_key,
        "parameters": {"max_new_tokens": 100},
    }

    inputs.append(payload["inputs"])
    ground_truth_responses.append(datapoint["response"])
    
    pretrained_response = pretrained_predictor.predict(payload, custom_attributes="accept_eula=true")
    responses_before_finetuning.append(pretrained_response[0]["generated_text"])
   
    finetuned_response = finetuned_predictor.predict(payload, custom_attributes="accept_eula=true")
    responses_after_finetuning.append(finetuned_response[0]["generated_text"])

try:
    for i, datapoint in enumerate(test_dataset.select(range(5))):
        predict_and_print(datapoint)

    df = pd.DataFrame(
        {
            "Inputs": inputs,
            "Ground Truth": ground_truth_responses,
            "Response from non-finetuned model": responses_before_finetuning,
            "Response from fine-tuned model": responses_after_finetuning,
        }
    )
    display(HTML(df.to_html()))
except Exception as e:
    print(e)


# ### Clean up resources

# In[ ]:


# Delete resources
pretrained_predictor.delete_model()
pretrained_predictor.delete_endpoint()
finetuned_predictor.delete_model()
finetuned_predictor.delete_endpoint()


# # What to do next
# - Understand the various parameters in finetuning
# - Calculate the costs for finetuning

# # Appendix
# 

# ### Supported Inference Parameters
# 
# ---
# This model supports the following inference payload parameters:
# 
# * **max_new_tokens:** Model generates text until the output length (excluding the input context length) reaches max_new_tokens. If specified, it must be a positive integer.
# * **temperature:** Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If `temperature` -> 0, it results in greedy decoding. If specified, it must be a positive float.
# * **top_p:** In each step of text generation, sample from the smallest possible set of words with cumulative probability `top_p`. If specified, it must be a float between 0 and 1.
# * **return_full_text:** If True, input text will be part of the output generated text. If specified, it must be boolean. The default value for it is False.
# 
# You may specify any subset of the parameters mentioned above while invoking an endpoint. 
# 
# 
# ### Notes
# - If `max_new_tokens` is not defined, the model may generate up to the maximum total tokens allowed, which is 4K for these models. This may result in endpoint query timeout errors, so it is recommended to set `max_new_tokens` when possible. For 7B, 13B, and 70B models, we recommend to set `max_new_tokens` no greater than 1500, 1000, and 500 respectively, while keeping the total number of tokens less than 4K.
# - In order to support a 4k context length, this model has restricted query payloads to only utilize a batch size of 1. Payloads with larger batch sizes will receive an endpoint error prior to inference.
# 
# ---

# ### Supported Hyper-parameters for fine-tuning
# ---
# - epoch: The number of passes that the fine-tuning algorithm takes through the training dataset. Must be an integer greater than 1. Default: 5
# - learning_rate: The rate at which the model weights are updated after working through each batch of training examples. Must be a positive float greater than 0. Default: 1e-4.
# - instruction_tuned: Whether to instruction-train the model or not. Must be 'True' or 'False'. Default: 'False'
# - per_device_train_batch_size: The batch size per GPU core/CPU for training. Must be a positive integer. Default: 4.
# - per_device_eval_batch_size: The batch size per GPU core/CPU for evaluation. Must be a positive integer. Default: 1
# - max_train_samples: For debugging purposes or quicker training, truncate the number of training examples to this value. Value -1 means using all of training samples. Must be a positive integer or -1. Default: -1. 
# - max_val_samples: For debugging purposes or quicker training, truncate the number of validation examples to this value. Value -1 means using all of validation samples. Must be a positive integer or -1. Default: -1. 
# - max_input_length: Maximum total input sequence length after tokenization. Sequences longer than this will be truncated. If -1, max_input_length is set to the minimum of 1024 and the maximum model length defined by the tokenizer. If set to a positive value, max_input_length is set to the minimum of the provided value and the model_max_length defined by the tokenizer. Must be a positive integer or -1. Default: -1. 
# - validation_split_ratio: If validation channel is none, ratio of train-validation split from the train data. Must be between 0 and 1. Default: 0.2. 
# - train_data_split_seed: If validation data is not present, this fixes the random splitting of the input training data to training and validation data used by the algorithm. Must be an integer. Default: 0.
# - preprocessing_num_workers: The number of processes to use for the preprocessing. If None, main process is used for preprocessing. Default: "None"
# - lora_r: Lora R. Must be a positive integer. Default: 8.
# - lora_alpha: Lora Alpha. Must be a positive integer. Default: 32
# - lora_dropout: Lora Dropout. must be a positive float between 0 and 1. Default: 0.05. 
# - int8_quantization: If True, model is loaded with 8 bit precision for training. Default for 7B/13B: False. Default for 70B: True.
# - enable_fsdp: If True, training uses Fully Sharded Data Parallelism. Default for 7B/13B: True. Default for 70B: False.
# 
# Note 1: int8_quantization is not supported with FSDP. Also, int8_quantization = 'False' and enable_fsdp = 'False' is not supported due to CUDA memory issues for any of the g5 family instances. Thus, we recommend setting exactly one of int8_quantization or enable_fsdp to be 'True'
# Note 2: Due to the size of the model, 70B model can not be fine-tuned with enable_fsdp = 'True' for any of the supported instance types.
# 
# ---

# ### Supported Instance types
# 
# ---
# We have tested our scripts on the following instances types:
# 
# - 7B: ml.g5.12xlarge, nl.g5.24xlarge, ml.g5.48xlarge, ml.p3dn.24xlarge
# - 13B: ml.g5.24xlarge, ml.g5.48xlarge, ml.p3dn.24xlarge
# - 70B: ml.g5.48xlarge
# 
# Other instance types may also work to fine-tune. Note: When using p3 instances, training will be done with 32 bit precision as bfloat16 is not supported on these instances. Thus, training job would consume double the amount of CUDA memory when training on p3 instances compared to g5 instances.
# 
# ---

# ### Few notes about the fine-tuning method
# 
# ---
# - Fine-tuning scripts are based on [this repo](https://github.com/facebookresearch/llama-recipes/tree/main). 
# - Instruction tuning dataset is first converted into domain adaptation dataset format before fine-tuning. 
# - Fine-tuning scripts utilize Fully Sharded Data Parallel (FSDP) as well as Low Rank Adaptation (LoRA) method fine-tuning the models
# 
# ---

# In[ ]:




