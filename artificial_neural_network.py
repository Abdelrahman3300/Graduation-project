import numpy as np
import pandas as pd
import pickle
#import tensorflow as tf
#tf.__version__

#from flask import Flask
#@app.route("/",methods=['GET'])
#app = Flask(__name__)
# if (__name__=="__main__")
# app.run()
     
#def ML() :
#
# series = c.execute("""SELECT * FROM sales()""")
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-2].values
y = dataset.iloc[:,-1].values
print(X)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 1] = le.fit_transform(X[:, 1])
print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#ann = tf.keras.models.Sequential()
#ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [100 -'accuracy'])
#ann.fit(X_train, y_train, batch_size = 32, epochs = 10)

# save the model to disk
#filename = 'finalized_model.sav'
#pickle.dump(ann, open(filename, 'wb'))

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test,y_test)
#print(result)