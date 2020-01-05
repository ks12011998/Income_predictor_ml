from stemming.porter2 import stem
import os
import re
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense,MaxPooling2D,BatchNormalization
from keras.models import load_model
import warnings
from django.core.cache import cache
from os import path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


url = "adult.csv"
df = pd.read_csv(url)

# filling missing values
col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", np.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#discretisation
df.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

#label Encoder
category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income'] 
labelEncoder = LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)

#droping redundant columns
df=df.drop(['fnlwgt','educational-num'], axis=1)

X = df.values[:, 0:12]
y = df.values[:,12]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 100)
dt_clf = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=20, min_samples_leaf=20)

dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)

print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test,y_pred)*100 )

#creating and training a model
#serializing our model to a file called model.pkl
import pickle
pickle.dump(dt_clf, open("model.pkl","wb"))

