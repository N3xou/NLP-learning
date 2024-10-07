import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

simple_train = ['call you tonight', 'Call me a cab', 'Please call me... PLEASE!']

vect = CountVectorizer()

# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(simple_train)

# examine the fitted vocabulary
vect.get_feature_names_out()

simple_train_dtm = vect.transform(simple_train)
simple_train_dtm
simple_train_dtm.toarray()