#!/usr/bin/env python
# coding: utf-8

# # Train a linear model for sentiment classification
# 

# # Development environment
# 

# In[1]:


# install packages
get_ipython().system(' pip install scikit-learn')
get_ipython().system(' pip install datasets')
get_ipython().system(' pip install wandb')
get_ipython().system(' pip install seaborn')


# In[2]:


import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import wandb
import time

import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Login to Weights and Biases
# 

# In[3]:


wandb.login()


# In[4]:


datetime = time.strftime("%Y%m%d-%H%M%S")
wandb.init(
      # Set the project where this run will be logged
      project="sutd-mlops-project", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_session2_run_{datetime}", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": 0.01,
      "loss": "log_loss",
      "penalty": "l2",
      "architecture": "SGDClassifier",
      "dataset_name": "rotten_tomatoes",
      })
config = wandb.config


# # Prepare data
# 

# In[17]:


# Download the Rotten Tomatoes dataset
dataset = load_dataset(config.dataset_name)

# print the first movie review and label
print(dataset["train"][0])


# In[18]:


dataset


# In[6]:


print(dataset["train"].column_names)


# In[7]:


labels = list(set(dataset['train']['label']))
print("Labels:", labels)


# In[24]:


dataset["train"].features['label']


# In[8]:


sns.countplot(x=dataset['train']['label'])
plt.xlabel('label');


# In[9]:


train_text = dataset['train']['text']
train_labels = dataset['train']['label']

test_text = dataset['test']['text']
test_labels = dataset['test']['label']


# In[10]:


count_vect = CountVectorizer()
train_features = count_vect.fit_transform(train_text)
test_features = count_vect.transform(test_text)


# # Train the model
# 

# In[11]:


model = SGDClassifier(
            loss = config.loss, 
            penalty = config.penalty,
            learning_rate = 'constant', 
            eta0 = config.learning_rate
        ).fit(train_features, train_labels)


# # Test the model
# 

# In[12]:


test_predicted = model.predict(test_features)
test_proba = model.predict_proba(test_features)
accuracy = metrics.accuracy_score(test_labels, test_predicted)
print(accuracy)


# In[13]:


wandb.log({"accuracy": accuracy})
wandb.sklearn.plot_precision_recall(test_labels, test_proba, ["negative", "positive"])



# In[14]:


wandb.finish()


# # What to try next
# 
# - experiment with different training parameters (iterations, learning rate, regulartization, ...)
# - experiment with different training set sizes
# - the dataset also has a validation set, what is the accuracy here?
# - use Weights & Biases plots to get more insights into the model behavior 
# 
