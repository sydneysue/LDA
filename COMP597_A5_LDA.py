# This script is for COMP597 - Assignment 5 -Sydney Sue
# LDA for EHRs

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from seaborn import heatmap

# %%

# Load data
ehr = pd.read_csv('/home/mcb/users/ssue1/comp597/DIAGNOSES_ICD_subset.csv')
meta = pd.read_csv('/home/mcb/users/ssue1/comp597/D_ICD_DIAGNOSES.csv')
# Remove leading zeros in ICD9_CODE
meta['ICD9_CODE'] = meta['ICD9_CODE'].str.lstrip('0')

# %%
# Hyper-parameters
topics = 5
alpha = 1
beta = 0.001
epochs = 100

# Assign ID to unique ICD9-code - tokenized long vector of ICD-9 indices
# Each ICD-9 observation is a 'token'
ehr['id'] = ehr.groupby(['ICD9_CODE']).ngroup()
# 3017 unique codes
vocab = ehr['id'].unique().tolist()

# Create a dictionary of the ID with the ICD9-codes
id_icd_dic = pd.Series(ehr.ICD9_CODE.values, index=ehr.id).to_dict()

# Each patient represents one 'document'
# 4011 unique subjects ie. documents
documents = ehr.groupby('SUBJECT_ID')['id'].apply(list).to_dict()

# %%
# word-topic matrix, word count of each topic and vocabulary
word_topic = np.zeros([topics,len(vocab)])
# topic-document matrix, word count of each document and topic
topic_doc = np.zeros([len(documents),topics])
# word count of each topic
word_count = np.zeros(topics)
n_d = np.zeros((len(documents)))
# topic for each word per document
word_doc = []

# For each ICD9 code for subject assign random topic
# Initialize parameters
for i, (k, v) in enumerate(documents.items()):
    # List of ICD9 code topics per subject
    z_subject = []
    for n, value in enumerate(v):
        z = np.random.randint(topics)
        z_subject.append(z)
        topic_doc[i, z] += 1
        word_topic[z, value] += 1
        word_count[z] += 1
        n_d[i] += 1
    word_doc.append(np.array(z_subject))
# %%
for epoch in range(epochs):
    for i, (k, v) in enumerate(documents.items()):
        for n, value in enumerate(v):
            # Get initial topic
            z = word_doc[i][n]


            # Don't include token w in matrix when sampling for token w
            topic_doc[i, z] -= 1
            word_topic[z, value] -= 1
            word_count[z] -= 1

            # Sample topic from a multinomial distribution
            prob_top_doc = (topic_doc[i] + alpha) / (np.sum(topic_doc[i]) + topics * alpha)
            prob_top_word = (word_topic[:, value] + beta) / (word_count + len(vocab) * beta)
            # Probability word belongs to topic
            p_z = prob_top_doc * prob_top_word
            # Draw new topic for word n from probability calculated above
            new_z = np.random.multinomial(1, (p_z/np.sum(p_z))).argmax()

            # Update new topic assignments
            word_doc[i][n] = new_z
            topic_doc[i, new_z] += 1
            word_topic[new_z, value] += 1
            word_count[new_z] += 1

    print("********** Finished epoch {} **********".format(epoch))

# %%
# Word distribution for topic (ie. probability of word per topic)

vocab_list = ehr['id'].unique().tolist()
topic_prob = (word_topic + beta) / np.sum(word_topic + beta, axis=1)[:,None]
topic_prob_df = pd.DataFrame(topic_prob).astype("float")
topic_prob_df.columns = vocab_list
topic_prob_df = topic_prob_df.T

# %%
# Get the top ten most probable IDs for each topic
top_ten = []

for topic in range (0,topics):
    x = topic_prob_df.nlargest(10,[topic])
    top_ten.append(x)

top_ten_df = pd.concat(top_ten)
print(top_ten_df)

# %%
# Concatenate ICD-9 topics
meta['Code_Title'] = meta['ICD9_CODE'] + ' - ' + meta['SHORT_TITLE']
meta.set_index('ICD9_CODE')
# Create a dictionary of ICD-9-codes and their short title
dic = pd.Series(meta.Code_Title.values,index=meta.ICD9_CODE).to_dict()

# Change the IDs in df to the corresponding ICD9-code
top_ten_df = top_ten_df.reset_index()
top_ten_df['index'] = top_ten_df['index'].map(id_icd_dic)

# Change the ICD9-Code to the corresponding short title
top_ten_df['index'] = top_ten_df['index'].astype(str).map(dic)
top_ten_df = top_ten_df.set_index('index')
# %% Plot heatmap
plt.figure(figsize=(10,20))
heatmap(top_ten_df, yticklabels=True, vmax=0.1, cmap="Reds")
plt.show()




