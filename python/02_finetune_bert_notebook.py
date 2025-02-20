#!/usr/bin/env python
# coding: utf-8

# # Finetune bert classifier for sentiment classification
# Example from https://huggingface.co/docs/transformers/training

# # Development environment
# 

# In[ ]:


get_ipython().system(' pip install -U transformers[torch]')
get_ipython().system(' pip install -U accelerate')
get_ipython().system(' pip install datasets')
get_ipython().system(' pip install evaluate')
get_ipython().system(' pip install scikit-learn')
get_ipython().system(' pip install wandb')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
import wandb
import time

import numpy as np
import evaluate


# # Login to Weights and Biases
# 

# In[ ]:


wandb.login()


# In[ ]:


wandb.init(
      # Set the project where this run will be logged
      project="sutd-mlops-project", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_session3_run_1", 
      # Track hyperparameters and run metadata
      config={
          "learning_rate": 2e-5,
          "weight_decay": 0.01,
          "num_train_epochs": 2,
          "train_subsample_size": 1000,
          "architecture": "distilbert",
          "dataset_name": "rotten_tomatoes",
          "model_name": "distilbert-base-uncased"
      })
config = wandb.config


# # Prepare data
# 

# In[ ]:


dataset = load_dataset(config.dataset_name)
dataset["train"][0]


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(config.model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


# In[ ]:


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(config.train_subsample_size))
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))


# # Train the model
# 

# In[ ]:


num_labels = len(np.unique(dataset['train']['label']))
model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=num_labels)


# In[ ]:


metric = evaluate.load("accuracy")


# In[ ]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# In[ ]:


training_args = TrainingArguments(
    output_dir=".",
    report_to="wandb",
    evaluation_strategy="epoch",
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    num_train_epochs=config.num_train_epochs,
    logging_steps=20)


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)


# In[ ]:


trainer.train()


# # Test the model
# 

# In[ ]:


# Accuracy on training set
trainer.evaluate(small_train_dataset)


# In[ ]:


# Accuracy on validation set
trainer.evaluate(small_eval_dataset)


# In[ ]:


# Accuracy on test set
trainer.evaluate(small_test_dataset)


# In[ ]:


# accuracy of the whole test set - for fair comparison with the classification performance achieved by SGD in previous sessions
def predict(tokenized_test_data, trainer):
    output_array = trainer.predict(tokenized_test_data)[0]
    pred_prob = np.exp(output_array)/np.sum(np.exp(output_array), axis = 1)[..., None]
    pred = np.argmax(pred_prob, axis = 1)
    return pred_prob, pred 

pred_prob, pred  = predict(tokenized_datasets["test"], trainer)
accuracy = np.sum(pred == dataset["test"]['label'])/len(dataset["test"]['label'])
print(f"Accuracy: {accuracy}")
wandb.sklearn.plot_precision_recall(dataset["test"]['label'], pred_prob, ["negative", "positive"])


# In[ ]:


wandb.finish()


# # What to try next
# 
# - train and evaluate with the complete training and test dataset instead of a sample
# - experiment with different training parameters (number of epochs, optimizers, batch size, learning rate schedule, ...)
# - compare DistilBERT vs the full BERT model: https://huggingface.co/bert-base-uncased
# - compare the results with the scikit model from the previous notebook. What is the cost-benefit trade off between deep learning and traditional ML?
# - Check out this more detailed sentiment tutorial on Huggingface https://huggingface.co/blog/sentiment-analysis-python

# In[ ]:




