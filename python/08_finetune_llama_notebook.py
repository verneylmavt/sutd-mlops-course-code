#!/usr/bin/env python
# coding: utf-8

# # Finetune OpenLlama in Sagemaker notebook 
# 
# In this notebook, we will perform instruction tuning OpenLlama using a subset of the Dolly 15k Dataset.
# 
# 
# This notebook as been put together based a few great examples and blogs. Feel free to visit them to learn more about finetuning. 
# 
# - [Fourthbrain Repository Building with Instruction-Tuned LLMs: a Step-by-Step Guide](https://github.com/FourthBrain/Building-with-Instruction-Tuned-LLMs-A-Step-by-Step-Guide)
# - [Notes on fine-tuning Llama 2 using QLoRA: A detailed breakdown. Blog by Ogban Ugot](https://medium.com/@ogbanugot/notes-on-fine-tuning-llama-2-using-qlora-a-detailed-breakdown-370be42ccca1)
# - [Interactively fine-tune Falcon-40B and other LLMs on Amazon SageMaker Studio notebooks using QLoRA. Blog by AWS](https://aws.amazon.com/blogs/machine-learning/interactively-fine-tune-falcon-40b-and-other-llms-on-amazon-sagemaker-studio-notebooks-using-qlora/)
# 

# ### ⚠ IMPORTANT ⚠
# 
# Please ensure your Jupyterlab instance is set to the following: **ml.g5.4xlarge**
# 

# # Development environment
# 
# We're going to be leveraging a number of awesome tools in order to be able to instruct-tune our model.
# 
# Here's a brief overview:
# 
# - [Hugging Face's PEFT Library](https://github.com/huggingface/peft)
# - [Hugging Face's Transformers Library](https://huggingface.co/docs/transformers/index)
# - [QLoRA](https://arxiv.org/abs/2305.14314)
# - [TRL](https://github.com/lvwerra/trl/tree/main/docs/source)
# 
# Keep in mind that these libraries are being constantly iterated on - and so you may experience bugs/issues.
# 

# In[ ]:


get_ipython().system(' pip install peft==0.4.0')
get_ipython().system(' pip install bitsandbytes==0.40.2')
get_ipython().system(' pip install transformers==4.31.0')
get_ipython().system(' pip install trl==0.4.7')
get_ipython().system(' pip install torch==2.0.1')
get_ipython().system(' pip install accelerate==0.21.0')
get_ipython().system(' pip install datasets')


# In[ ]:


# Add installed cuda runtime to path for bitsandbytes
import os
import nvidia

cuda_install_dir = '/'.join(nvidia.__file__.split('/')[:-1]) + '/cuda_runtime/lib/'
os.environ['LD_LIBRARY_PATH'] =  cuda_install_dir


# In[ ]:


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


# # Prepare dataset

# Let's look at our dataset to get an idea of what we're working with!

# In[ ]:


from datasets import load_dataset

dbricks_15k_dataset_base = load_dataset("databricks/databricks-dolly-15k")


# Let's check out some brief stats about our dataset:

# In[ ]:


import matplotlib.pyplot as plt
from datasets import load_dataset

def plot_and_filter_sequence_lengths(dataset_obj, max_length=2200):

    # Initialize a list to store the sequence lengths
    sequence_lengths = []

    # list of indices that are too long
    too_long = []

    # Loop over the dataset and get the lengths of text sequences
    for idx, example in enumerate(dataset_obj["train"]):
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


indexes_to_drop = plot_and_filter_sequence_lengths(dbricks_15k_dataset_base, max_length=1000)


# In[ ]:


len(indexes_to_drop)


# In[ ]:


dbricks_15k_dataset_reduced = dbricks_15k_dataset_base["train"].select(
    i for i in range(len(dbricks_15k_dataset_base["train"])) if i not in set(indexes_to_drop)
)


# In[ ]:


dbricks_15k_dataset_reduced


# In[ ]:


dbricks_15k_dataset_prepared = dbricks_15k_dataset_reduced.train_test_split(test_size=0.1)


# In[ ]:


plot_and_filter_sequence_lengths(dbricks_15k_dataset_prepared)


# In[ ]:


dbricks_15k_dataset_prepared


# Before we can begin training, we need to set up a few helper functions to ensure our dataset is parsed in the correct format and we save our PEFT adapters!

# In[ ]:


def formatting_func(example):
  if example.get("context", "") != "":
      input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Input: \n"
      f"{example['context']}\n\n"
      f"### Response: \n"
      f"{example['response']}")

  else:
    input_prompt = (f"Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Response:\n"
      f"{example['response']}")

  return {"text" : input_prompt}


# In[ ]:


formatted_dataset = dbricks_15k_dataset_prepared.map(formatting_func)


# In[ ]:


formatted_dataset


# In[ ]:


import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(formatted_dataset["train"][0])


# Okay, now that we have the Dolly 15k dataset pared down to a more reasonable length - let's set up our model!
# 
# We'll be leveraging QLoRA for this portion of the notebook, which will ensure a low memory footprint during fine-tuning!
# 
# - [Paper](https://arxiv.org/pdf/2305.14314.pdf)
# - [Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

# # Finetune the model

# In[ ]:


# The model that you want to train from the Hugging Face hub
model_name = "openlm-research/open_llama_3b_v2"

# Fine-tuned model name
new_model = "open_llama_3b_v2_chat_dolly"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}


# Now, let's set up our SupervisedFineTuningTrainer and let it rip!
# 
# More information on the SFTTrainer is available here:
# 
# - [HF Documentation](https://huggingface.co/docs/trl/main/en/sft_trainer)
# - [Repository](https://github.com/lvwerra/trl/blob/main/trl/trainer/sft_trainer.py)
# 
# 

# In[ ]:


# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)


# In[ ]:


# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)


# # Test model

# In[ ]:


#evaluate and return the metrics
trainer.evaluate()


# In[ ]:


# Empty VRAM
del model
del trainer
import gc
gc.collect()
gc.collect()


# In[ ]:


# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# In[ ]:


from huggingface_hub import notebook_login

notebook_login()


# In[ ]:


model.push_to_hub(new_model, use_temp_dir=False)


# In[ ]:


tokenizer.push_to_hub(new_model, use_temp_dir=False)


# In[ ]:


from peft import get_peft_model
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import AutoTokenizer

lora_config = LoraConfig.from_pretrained(new_model)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(new_model)
model = AutoModelForCausalLM.from_pretrained(
    lora_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map={"":0})


# In[ ]:


model = get_peft_model(model, lora_config)


# In[ ]:


from IPython.display import display, Markdown

def make_inference(instruction, context = None):
  if context:
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction: \n{instruction}\n\n### Input: \n{context}\n\n### Response: \n"
  else:
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: \n{instruction}\n\n### Response: \n"
  inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:0")
  outputs = base_model.generate(**inputs, max_new_tokens=100)
  print("### Basemodel")
  display(Markdown((tokenizer.decode(outputs[0], skip_special_tokens=True))))
  outputs = model.generate(**inputs, max_new_tokens=100)
  print("### Finetuned model")
  display(Markdown((tokenizer.decode(outputs[0], skip_special_tokens=True))))


# In[ ]:


make_inference("Explain the moon landing to a 5 year old.")


# In[ ]:


make_inference("Identify the odd one out and explain your choice.", "Orange, Green, Airplane.")


# In[ ]:


make_inference("When was Kyoto the capital of Japan?", "Kyoto is one of the oldest municipalities in Japan, having been chosen in 794 as the new seat of Japan's imperial court by Emperor Kanmu. The original city, named Heian-kyō, was arranged in accordance with traditional Chinese feng shui following the model of the ancient Chinese capitals of Chang'an and Luoyang. The emperors of Japan ruled from Kyoto in the following eleven centuries until 1869. It was the scene of several key events of the Muromachi period, Sengoku period, and the Boshin War, such as the Ōnin War, the Honnō-ji Incident, the Kinmon incident and the Battle of Toba–Fushimi. The capital was relocated from Kyoto to Tokyo after the Meiji Restoration.")


# In[ ]:


make_inference("Where is Hoober Stand located?", "Hoober Stand is a 30-metre-high (98 ft) tower and Grade II* listed building on a ridge in Wentworth, South Yorkshire in northern England. It was designed by Henry Flitcroft for the Whig aristocrat Thomas Watson-Wentworth, Earl of Malton (later the 1st Marquess of Rockingham) to commemorate the quashing of the 1745 Jacobite rebellion. It lies close to his country seat Wentworth Woodhouse. Its site is approximately 157 metres (515 ft) above sea level and from the top there are long-distance views on a clear day. Hoober Stand is one of several follies in and around Wentworth Woodhouse park; the others include Needle's Eye and Keppel's Column. Sidney Oldall Addy, the Sheffield author calls the structure Woburn Stand in his 1888 book, A glossary of words used in the neighbourhood of Sheffield.")


# In[ ]:


make_inference("Given this paragraph about hedgehogs, why are they different from porcupines?", "Hedgehogs are easily recognized by their spines, which are hollow hairs made stiff with keratin.Their spines are not poisonous or barbed and, unlike the quills of a porcupine, do not easily detach from their bodies. However, the immature animal's spines normally fall out as they are replaced with adult spines. This is called \"quilling\". Spines can also shed when the animal is diseased or under extreme stress. Hedgehogs are usually brown, with pale tips to the spines, though blonde hedgehogs are found on the Channel Island of Alderney.")


# In[ ]:


make_inference("Given this paragraph, who wrote and directed Heads I Win, Tails You Lose", "Heads I Win, Tails You Lose (Italian: Testa o Croce, also known as Heads or Tails) is a 1982 Italian comedy film written and directed by Nanni Loy.\n\nThe film consists in two back-to-back stories that deals with two \"taboo\" themes, the celibacy of the clergy in the episode of Renato Pozzetto and the homosexuality in the one with Nino Manfredi.")


# In[ ]:


make_inference("Why is free climbing called free climbing", "Most of the climbing done in modern times is considered free climbing—climbing using one's own physical strength, with equipment used solely as protection and not as support—as opposed to aid climbing, the gear-dependent form of climbing that was dominant in the sport's earlier days. Free climbing is typically divided into several styles that differ from one another depending on the choice of equipment used and the configurations of their belay, rope and anchor systems")


# # What to do next
# - Understand the finetuning parameters
# - Is there any difference to the untrained model?
# - How well is ChatGPT doing on this task?

# In[ ]:




