#In[ ]:
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# In[ ]: lectura de csv
df = pd.read_csv("C:\\Users\\yo\\Dropbox\\Pruebas challenge\\TRAIN_set.csv", sep=",")
df.drop(columns='user')
X_pre = df.iloc[:,:-1].values 
y_pre = df.iloc[:,-1]
y_pre_dum= pd.get_dummies(y_pre).values
X = []
y = []
# In[ ]: modificacion dimension de dataset
for j in range (X_pre.shape[0]):   #row
    for i in range (200):    #column
        X.append([X_pre[j,i], X_pre[j,i+200], X_pre[j, i+400]])
data = np.array(X)
X=data
y=y_pre_dum      
# In[ ]: generacion de LSTM y prediccion
model_lstm = Sequential()
model_lstm.add(LSTM(300, input_shape=(200,3), return_sequences=False))
model_lstm.add(Dense(4, activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
X = X.reshape(-1, 200, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model_lstm.fit(X_train, y_train, batch_size=4, epochs=50)
y_pred = model_lstm.predict_classes(X_test)

# In[ ]: parametros de test
y_test_df = pd.DataFrame({0:y_test[:,0],1:y_test[:,1],2:y_test[:,2],3:y_test[:,3]})
y_test_df_idx=y_test_df.idxmax(axis=1)

print(confusion_matrix(y_test_df_idx, y_pred))
print(classification_report(y_test_df_idx, y_pred))
print('Accuracy %0.4f' % accuracy_score(y_test_df_idx, y_pred))
 # In[ ]: generacion de predicciones con datos de test finales
 
df_test = pd.read_csv("C:\\Users\\yo\\Dropbox\\Pruebas challenge\\INSE1819_challenge_TEST.csv", sep=",")
df_test = df_test.drop(columns='user')
X_pre = df_test.iloc[:,:-1].values 
y_pre = df_test.iloc[:,-1]

X = []
y = []
for j in range (X_pre.shape[0]):   #row
    for i in range (200):    #column
        X.append([X_pre[j,i], X_pre[j,i+200], X_pre[j, i+400]])
data = np.array(X)
X=data 
X_test_f= X.reshape(-1, 200, 3)
y_pred = model_lstm.predict(X_test_f)
y_test_dff = pd.DataFrame({'BACKHAND':y_pred[:,0],'DRIVE':y_pred[:,1],'LOB':y_pred[:,2],'SERVE':y_pred[:,3]})
y_test_dff_idx=y_test_dff.idxmax(axis=1)

df_test_out = pd.DataFrame()
df_test_out['class_prev'] = df_test['class']
df_test_out['class_pred'] = y_test_dff_idx

df_test_out.to_csv('test_predictions_INSE1819.csv', sep=',', header=True)
