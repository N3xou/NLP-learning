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