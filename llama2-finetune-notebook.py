#!/usr/bin/env python
# coding: utf-8

# # Finetune Llama-2 7B in Sagemaker notebook 
# 
# From https://aws.amazon.com/blogs/machine-learning/interactively-fine-tune-falcon-40b-and-other-llms-on-amazon-sagemaker-studio-notebooks-using-qlora/
# and https://medium.com/@ogbanugot/notes-on-fine-tuning-llama-2-using-qlora-a-detailed-breakdown-370be42ccca1
# 

# In[ ]:


get_ipython().run_line_magic('pip', 'install -q -U torch==2.0.1 bitsandbytes==0.40.2')
get_ipython().run_line_magic('pip', 'install -q -U transformers==4.31.0 peft==0.4.0 accelerate==0.21.0')
get_ipython().run_line_magic('pip', 'install -q -U datasets py7zr einops tensorboardX')


# In[ ]:


# Add installed cuda runtime to path for bitsandbytes
import os
import nvidia

cuda_install_dir = '/'.join(nvidia.__file__.split('/')[:-1]) + '/cuda_runtime/lib/'
os.environ['LD_LIBRARY_PATH'] =  cuda_install_dir


# In[ ]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "NousResearch/Llama-2-7b-hf" # not gated
bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")


# In[ ]:


from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# In[ ]:


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# In[ ]:


from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


# In[ ]:


from datasets import load_dataset

# Load dataset from the hub, load subset of training data for faster training
train_dataset = load_dataset("samsum", split="train[:10%]")
test_dataset = load_dataset("samsum", split="test")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# In[ ]:


from random import randint

# custom instruct prompt start
prompt_template = f"Summarize the chat dialogue:\n{{dialogue}}\n---\nSummary:\n{{summary}}{{eos_token}}"

# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = prompt_template.format(dialogue=sample["dialogue"],
                                            summary=sample["summary"],
                                            eos_token=tokenizer.eos_token)
    return sample


# apply prompt template per sample
train_dataset = train_dataset.map(template_dataset, remove_columns=list(train_dataset.features))

print(train_dataset[randint(0, len(train_dataset))]["text"])


# In[ ]:


# apply prompt template per sample
test_dataset = test_dataset.map(template_dataset, remove_columns=list(test_dataset.features))


# In[ ]:


# tokenize and chunk dataset
lm_train_dataset = train_dataset.map(
   lambda sample: tokenizer(sample["text"]), batched=True, batch_size=24, remove_columns=list(train_dataset.features)
)

lm_test_dataset = test_dataset.map(
   lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(test_dataset.features)
)

# Print total number of samples
print(f"Total number of train samples: {len(lm_train_dataset)}")


# In[ ]:


import transformers

# We set num_train_epochs=1 simply to run a demonstration

trainer = transformers.Trainer(
    model=model,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_test_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        output_dir="outputs"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inferenc


# In[ ]:


# Start training
trainer.train()


# In[ ]:


#evaluate and return the metrics
trainer.evaluate()


# Load test dataset and try a random sample for summarization

# In[ ]:


# Load dataset from the hub
test_dataset = load_dataset("samsum", split="test")

# select a random test sample
sample = test_dataset[randint(0, len(test_dataset))]

# format sample
prompt_template = f"Summarize the chat dialogue:\n{{dialogue}}\n---\nSummary:\n"

test_sample = prompt_template.format(dialogue=sample["dialogue"])

print(test_sample)


# In[ ]:


input_ids = tokenizer(test_sample, return_tensors="pt").input_ids

#set the token length for the summary evaluation
tokens_for_summary = 30
output_tokens = input_ids.shape[1] + tokens_for_summary
outputs = model.generate(inputs=input_ids, do_sample=True, max_length=output_tokens)
gen_text = tokenizer.batch_decode(outputs)[0]
print(gen_text)


# In[ ]:





# # What to do next
# - Understand the finetuning parameters
# - Is there any difference to the untrained model?
# - How well is ChatGPT doing on this task?
