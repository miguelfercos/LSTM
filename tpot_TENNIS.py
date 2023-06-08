import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# lectura de CSV
df = pd.read_csv("C:\\Users\\yo\\Dropbox\\Pruebas challenge\\TRAIN_set.csv", sep=",")
df['user'] = df['user'].astype(str)
y = df['class'].values
df.pop('class')
#preprocesado y division entre datos de test y de train
#df = pd.get_dummies(df, prefix=['user'])
df=df.drop(columns='user')
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)
#aplicacion de modelo, entrenamiento y prediccion
model = ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.35, min_samples_leaf=1, min_samples_split=9, n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#obtencion de parametros
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))