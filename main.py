import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

simple_train = ['call you tonight', 'Call me a cab', 'Please call me... PLEASE!']

vect = CountVectorizer()

vect.fit(simple_train)

print(vect.get_feature_names_out())

simple_train_dtm = vect.transform(simple_train)
simple_train_dtm
simple_train_dtm.toarray()

pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())

print(type(simple_train_dtm))

print(simple_train_dtm)

# example text for model testing
simple_test = ["please don't call me"]

simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()
# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names_out())

# read file into pandas using a relative path
sms = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')
sms.dropna(how="any", inplace=True, axis=1)
sms.columns = ['label', 'message']

sms.head()

sms.describe()

sms.groupby('label').describe()

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
sms.head()

sms['message_len'] = sms.message.apply(len)
sms.head()

plt.figure(figsize=(12, 8))

sms[sms.label=='ham'].message_len.plot(bins=35, kind='hist', color='blue',
                                       label='Ham messages', alpha=0.6)
sms[sms.label=='spam'].message_len.plot(kind='hist', color='red',
                                       label='Spam messages', alpha=0.6)
plt.legend()
plt.xlabel("Message Length")

sms[sms.label=='ham'].describe()

sms[sms.label=='spam'].describe()

sms[sms.message_len == 910].message.iloc[0]