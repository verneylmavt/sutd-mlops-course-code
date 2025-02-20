#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install datasets')
get_ipython().system(' pip install scikit-learn')
get_ipython().system(' pip install seaborn')


# In[ ]:


from datasets import load_dataset
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import numpy as np



import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Download the Rotten Tomatoes dataset
rotten_tomatoes_dataset = load_dataset("rotten_tomatoes")

# select random 20 samples of movie review data 
text_list = rotten_tomatoes_dataset['train'].shuffle(seed=42)[0:20]


# In[ ]:


import json
from os import listdir
from os.path import isfile, join

output_dir = "label-output"
manifest_files = [f for f in listdir(output_dir) if isfile(join(output_dir, f))]
print(manifest_files)


# In[ ]:


# normalize labels to 0, 1
map_label = {"0": 0,
             "1": 1,
             "positive": 1,
             "negative": 0,
             "Positive": 1,
             "Negative": 0,
             "pOsiTive": 1,
             "nEgaTiVE": 0,
             "PAWsitive": 1,
             "NAGitive": 0,
             "pos": 1,
             "neg": 0,
             "2": 1,            # one label set had label "2", just assuming it means positive
            }


def add_groundtruth_output(groundtruth_items, combined_labels):
    for groundtruth_item in groundtruth_items.splitlines():
        print(groundtruth_item)
        if groundtruth_item.strip() == "":
            continue
        groundtruth_item = json.loads(groundtruth_item)
        text = groundtruth_item['source']
        label_job = list(filter(lambda x: x.endswith("metadata"), groundtruth_item.keys()))[0]
        label = map_label[groundtruth_item[label_job]['class-name']]
        combined_labels[text]["human_labels"].append(label)
        
# create dictionary with review text as key and gold label as value
combined_labels = dict(zip(text_list['text'], map(lambda x: {"gold_label": x, "human_labels": []}, text_list['label'])))

for manifest_file in manifest_files:
    with open(join(output_dir, manifest_file),'r') as f:
        add_groundtruth_output(f.read(), combined_labels)
    

combined_labels    
    


# In[ ]:


from collections import Counter
from operator import itemgetter

labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]

def majority_vote(labels):
    majority_label = max(Counter(labels).items(), key=itemgetter(1))[0]
    return majority_label


for text, label_dict in combined_labels.items():
    label_dict["majority_label"] = majority_vote(label_dict["human_labels"])

combined_labels
    


# In[ ]:


# compute accuracy, p/r/f1
gold_labels = [item["gold_label"]  for item in combined_labels.values()]
majority_labels = [item["majority_label"]  for item in combined_labels.values()]

print(classification_report(gold_labels, majority_labels))


# In[ ]:


# compute cohen's kappa
print(cohen_kappa_score(gold_labels, majority_labels))


# In[ ]:


# review counter 
review_id = 0

# show review, gold labels, majority vote, and distribution plot
review_text, label_dict = list(combined_labels.items())[review_id]

print("Review:", review_text)
print("Gold label:", label_dict["gold_label"])
print("Majority label:", label_dict["majority_label"])
print("Human labels:", label_dict["human_labels"])
print("Variance:", np.var(label_dict["human_labels"]))

sns.countplot(x=label_dict["human_labels"])
plt.xlabel('label');

