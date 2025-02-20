#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install datasets')


# In[ ]:


from datasets import load_dataset


# In[ ]:


# Download the Rotten Tomatoes dataset
rotten_tomatoes_dataset = load_dataset("rotten_tomatoes")

# print the first movie review and label
print(rotten_tomatoes_dataset["train"][0])


# In[ ]:


# select random 20 samples of movie review data 
text_list = rotten_tomatoes_dataset['train'].shuffle(seed=42)[0:20]



# In[ ]:


text_list


# In[ ]:


# create directory
get_ipython().system(' mkdir -p text_sample_20')


# In[ ]:


import sagemaker
import boto3


# create bucket
s3 = boto3.resource('s3')
bucket = s3.Bucket('test03-sagemaker-groundtruth')  # you need to change the bucket name

# save into txt file and upload to S3
for k, sample_text in enumerate(text_list['text']):
    with open(f'text_sample_20/{k}.txt', 'w') as f:
        f.write(sample_text)
    bucket.upload_file(f'text_sample_20/{k}.txt', f'rotten_tomatoes/text_sample_20/{k}.txt')


# In[ ]:


import json

with open("output.manifest",'r') as f:
    data20 = f.read()


# In[ ]:


temp = data20.split('\n')
temp


# In[ ]:


import json
labeled_data={
    'text':[],
    'label':[]
}   # Dictionary to save text, label from each object

# normalize labels to 0, 1
map_label = {"0": 0,
             "1": 1,
             "positive": 1,
             "negative": 0
            }

for obj in temp:
    if obj == "":
        continue
    temp_dict = json.loads(obj)    #Dictionary for 20 labeled sample
    labeled_data['text'].append(temp_dict['source'])   # Each object's text
    
    # find out what the label job key is
    label_job = list(filter(lambda x: x.endswith("metadata"), temp_dict.keys()))[0]
    labeled_data['label'].append(map_label[temp_dict[label_job]['class-name']])


# In[ ]:


labeled_data


# In[ ]:


gold_labels = []
human_labels = []

# align labels on the same review text
for o_text,o_label in zip(text_list['text'],text_list['label']):
    for l_text,l_label in zip(labeled_data['text'],labeled_data['label']):
        if o_text == l_text:
            print(f'Original text is:{o_text}')
            print(f"Sample text for labeling:{l_text}")
            print(f"Original label is:{o_label},sample label is :{l_label}")
            gold_labels.append(o_label)
            human_labels.append(l_label)
            print("---")


# In[ ]:


# compute accuracy, p/r/f1

from sklearn.metrics import classification_report

print(classification_report(gold_labels, human_labels))


# In[ ]:


# compute cohen's kappa

from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(gold_labels, human_labels)

