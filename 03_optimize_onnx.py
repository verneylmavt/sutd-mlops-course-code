#!/usr/bin/env python
# coding: utf-8

# # End-to-End tutorial on accelerating RoBERTa for Question-Answering including quantization and optimization
# From https://github.com/huggingface/blog/blob/main/optimum-inference.md
# 
# In this End-to-End tutorial on accelerating RoBERTa for question-answering, you will learn how to:
# 
# 1. Install Optimum for ONNX Runtime
# 2. Convert a Hugging Face Transformers model to ONNX for inference
# 3. Use the ORTOptimizer to optimize the model
# 4. Use the ORTQuantizer to apply dynamic quantization
# 5. Run accelerated inference using Transformers pipelines
# 6. Evaluate the performance and speed
#    
# Let’s get started 🚀
# 
# This tutorial was created and run on an m5.xlarge AWS EC2 Instance and also works on the SUTD cluster.

# ## Install Optimum for Onnxruntime
# Our first step is to install Optimum with the onnxruntime utilities.
# 

# In[ ]:


get_ipython().system('pip install optimum[onnxruntime]')



# ## 3.2 Convert a Hugging Face Transformers model to ONNX for inference
# Before we can start optimizing we need to convert our vanilla transformers model to the onnx format. The model we are using is the deepset/roberta-base-squad2 a fine-tuned RoBERTa model on the SQUAD2 question answering dataset.

# In[ ]:


from pathlib import Path
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

model_id = "deepset/roberta-base-squad2"
onnx_path = Path("onnx")
task = "question-answering"

# load vanilla transformers and convert to onnx
model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# save onnx checkpoint and tokenizer
model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)

# test the model with using transformers pipeline, with handle_impossible_answer for squad_v2
optimum_qa = pipeline(task, model=model, tokenizer=tokenizer, handle_impossible_answer=True)
prediction = optimum_qa(question="What's my name?", context="My name is Philipp and I live in Nuremberg.")

print(prediction)
# {'score': 0.9041663408279419, 'start': 11, 'end': 18, 'answer': 'Philipp'}


# We successfully converted our vanilla transformers to onnx and used the model with the transformers.pipelines to run the first prediction. Now let's optimize it.

# ## Use the ORTOptimizer to optimize the model
# After we saved our onnx checkpoint to onnx/ we can now use the ORTOptimizer to apply graph optimization, such as operator fusion and constant folding to accelerate latency and inference.

# In[ ]:


from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

# create ORTOptimizer and define optimization configuration
optimizer = ORTOptimizer.from_pretrained(onnx_path)

optimization_config = OptimizationConfig(optimization_level=99) # enable all optimizations

# apply the optimization configuration to the model
optimizer.optimize(save_dir=onnx_path, optimization_config=optimization_config)


# To test performance we can use the ORTModelForQuestionAnswering class again and provide an additional file_name parameter to load our optimized model. 

# In[ ]:


from optimum.onnxruntime import ORTModelForQuestionAnswering

# load optimized model
opt_model = ORTModelForQuestionAnswering.from_pretrained(onnx_path, file_name="model_optimized.onnx")

# test the quantized model with using transformers pipeline
opt_optimum_qa = pipeline(task, model=opt_model, tokenizer=tokenizer, handle_impossible_answer=True)
prediction = opt_optimum_qa(question="What's my name?", context="My name is Philipp and I live in Nuremberg.")
print(prediction)
# {'score': 0.9041661620140076, 'start': 11, 'end': 18, 'answer': 'Philipp'}


# ## Use the ORTQuantizer to apply dynamic quantization
# Another option to reduce model size and accelerate inference is by quantizing the model using the ORTQuantizer.

# In[ ]:


from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# create ORTQuantizer and define quantization configuration
quantizer = ORTQuantizer.from_pretrained(onnx_path, file_name="model.onnx")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

# apply the quantization configuration to the model
quantizer.quantize(save_dir=onnx_path, quantization_config=qconfig)


# We can now compare this model size as well as some latency performance
# 
# 

# In[ ]:


import os
# get model file size
size = os.path.getsize(onnx_path / "model.onnx")/(1024*1024)
print(f"Vanilla Onnx Model file size: {size:.2f} MB")
size = os.path.getsize(onnx_path / "model_quantized.onnx")/(1024*1024)
print(f"Quantized Onnx Model file size: {size:.2f} MB")

# Vanilla Onnx Model file size: 473.51 MB
# Quantized Onnx Model file size: 119.15 MB


# ## Run accelerated inference using pipelines
# 
# Optimum has built-in support for transformers pipelines. This allows us to leverage the same API that we know from using PyTorch and TensorFlow models.

# In[ ]:


from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

quant_model = ORTModelForQuestionAnswering.from_pretrained(onnx_path, file_name="model_quantized.onnx")

quantized_optimum_qa = pipeline("question-answering", model=quant_model, tokenizer=tokenizer)
prediction = quantized_optimum_qa(question="What's my name?", context="My name is Philipp and I live in Nuremberg.")

print(prediction)
# {'score': 0.806605339050293, 'start': 11, 'end': 18, 'answer': 'Philipp'}


# In addition to this optimum has a pipelines API which guarantees more safety for your accelerated models.

# In[ ]:


from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForQuestionAnswering
from optimum.pipelines import pipeline

tokenizer = AutoTokenizer.from_pretrained(onnx_path, file_name="model_quantized.onnx")
quant_model = ORTModelForQuestionAnswering.from_pretrained(onnx_path, file_name="model_quantized.onnx")
                                                     
quantized_optimum_qa = pipeline("question-answering", model=quant_model, tokenizer=tokenizer, handle_impossible_answer=True)
prediction = quantized_optimum_qa(question="What's my name?", context="My name is Philipp and I live in Nuremberg.")

print(prediction)
# {'score': 0.806605339050293, 'start': 11, 'end': 18, 'answer': 'Philipp'}


# ## Evaluate performance and speed
# As the last step, we want to take a detailed look at the performance and accuracy of our model. Applying optimization techniques, like graph optimizations or quantization not only impact performance (latency) those also might have an impact on the accuracy of the model. So accelerating your model comes with a trade-off.
# 
# Let's evaluate our models. Our transformers model deepset/roberta-base-squad2 was fine-tuned on the SQUAD2 dataset. This will be the dataset we use to evaluate our models.
# To safe time, we only load 10% of the dataset.

# In[ ]:


from datasets import load_metric,load_dataset

# load 10% of the data to safe time
metric = load_metric("squad_v2")
dataset = load_dataset("squad_v2", split="validation[:10%]")

print(f"length of dataset {len(dataset)}")
#length of dataset 1187


# In[ ]:


def evaluate(example):
  default = optimum_qa(question=example["question"], context=example["context"])
  optimized = opt_optimum_qa(question=example["question"], context=example["context"])
  quantized = quantized_optimum_qa(question=example["question"], context=example["context"])
  return {
      'reference': {'id': example['id'], 'answers': example['answers']},
      'default': {'id': example['id'],'prediction_text': default['answer'], 'no_answer_probability': 0.},
      'optimized': {'id': example['id'],'prediction_text': optimized['answer'], 'no_answer_probability': 0.},
      'quantized': {'id': example['id'],'prediction_text': quantized['answer'], 'no_answer_probability': 0.},
      }

result = dataset.map(evaluate)


# Now lets compare the results
# 
# 

# In[ ]:


default_acc = metric.compute(predictions=result["default"], references=result["reference"])
optimized = metric.compute(predictions=result["optimized"], references=result["reference"])
quantized = metric.compute(predictions=result["quantized"], references=result["reference"])

print(f"vanilla model: exact={default_acc['exact']}% f1={default_acc['f1']}%")
print(f"optimized model: exact={quantized['exact']}% f1={optimized['f1']}%")
print(f"quantized model: exact={quantized['exact']}% f1={quantized['f1']}%")

# vanilla model: exact=81.12889637742207% f1=83.27089343306695%
# quantized model: exact=80.6234203875316% f1=82.86541222514259%


# The quantized model achived an exact match of 80.62% and an f1 score of 82.86% which is 99% of the original model.
# 
# Okay, let's test the performance (latency) of our optimized and quantized model.
# 
# But first, let’s extend our context and question to a more meaningful sequence length of 128.

# In[ ]:


context="Hello, my name is Philipp and I live in Nuremberg, Germany. Currently I am working as a Technical Lead at Hugging Face to democratize artificial intelligence through open source and open science. In the past I designed and implemented cloud-native machine learning architectures for fin-tech and insurance companies. I found my passion for cloud concepts and machine learning 5 years ago. Since then I never stopped learning. Currently, I am focusing myself in the area NLP and how to leverage models like BERT, Roberta, T5, ViT, and GPT2 to generate business value."
question="As what is Philipp working?"


# To keep it simple, we are going to use a python loop and calculate the avg/mean latency for our vanilla model and for the optimized and quantized model.
# 
# 

# In[ ]:


from time import perf_counter
import numpy as np

def measure_latency(pipe):
    latencies = []
    # warm up
    for _ in range(10):
        _ = pipe(question=question, context=context)
    # Timed run
    for _ in range(100):
        start_time = perf_counter()
        _ =  pipe(question=question, context=context)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    return f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}"

print(f"Vanilla model {measure_latency(optimum_qa)}")
print(f"Optimized model {measure_latency(opt_optimum_qa)}")
print(f"Quantized model {measure_latency(quantized_optimum_qa)}")

# Vanilla model Average latency (ms) 102
# Optimized model Average latency (ms) 101
# Quantized model Average latency (ms) 46


# We managed to reduce our model latency from 102ms to 46ms or by 55%, while keeping 99% of the accuracy. 

# In[ ]:




