# In[ ]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier

df = pd.read_csv("C:\\Users\\yo\\Dropbox\\Pruebas challenge\\INSE1819_challenge_TRAIN.csv", sep=",")
df=df.drop(columns='user')
X = df.iloc[:,:-1].values 
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# In[ ]:
model = DummyClassifier(strategy='stratified', random_state=None, constant=None)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy %0.4f' % accuracy_score(y_test, y_pred))
scores = cross_val_score(model, X, y, cv=10)
print ('Accuracy  with 10-fold CV %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() *2))
# In[ ]:
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy %0.4f' % accuracy_score(y_test, y_pred))
scores = cross_val_score(model, X, y, cv=10)
print ('Accuracy  with 10-fold CV %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() *2))
# In[ ]:
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy %0.4f' % accuracy_score(y_test, y_pred))
scores = cross_val_score(model, X, y, cv=10)
print ('Accuracy  with 10-fold CV %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() *2))
# In[ ]:
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy %0.4f' % accuracy_score(y_test, y_pred))
scores = cross_val_score(model, X, y, cv=10)
print ('Accuracy  with 10-fold CV %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() *2))
# In[ ]:Omitida por largo tiempo de procesamiento
model = svm.SVC(kernel='linear')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy %0.4f' % accuracy_score(y_test, y_pred))
scores = cross_val_score(model, X, y, cv=10)
print ('Accuracy  with 10-fold CV %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() *2))
# In[ ]:
#tpot = TPOTClassifier(generations=5,verbosity=2)
tpot = TPOTClassifier(generations=10, population_size=50, scoring='accuracy', cv=5, max_time_mins=60, max_eval_time_mins=0.04, early_stop=2, n_jobs=2,verbosity=2)
tpot.fit(X_train,y_train)
print('Accuracy %0.4f' % accuracy_score(y_test, y_pred))
scores = cross_val_score(tpot, X, y, cv=10)
print ('Accuracy  with 10-fold CV %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() *2))

tpot.export('C:\\Users\\yo\\Dropbox\\Pruebas challenge\\tpot_no_user_train.py')