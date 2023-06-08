#In[ ]:
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# In[ ]: lectura de dataset y preprocesado
df = pd.read_csv("C:\\Users\\yo\\Dropbox\\Pruebas challenge\\TRAIN_set.csv", sep=",")
df_dum=pd.get_dummies(df, prefix=['user'])
X_pre = df_dum.iloc[:,:-1].values 
y_pre = df.iloc[:,-1]
y_pre_dum= pd.get_dummies(y_pre).values
X = []
y = []
# In[ ]: conversion de datos
for j in range (X_pre.shape[0]):   #row
    for i in range (200):    #column
        X.append([X_pre[j,i], X_pre[j,i+200], X_pre[j, i+400], X_pre[j, 600],  X_pre[j, 601], X_pre[j, 602], X_pre[j, 603]])
data = np.array(X)
X=data
y=y_pre_dum      
# In[ ]: 
model = Sequential()
model.add(LSTM(256, input_shape=(200,7), return_sequences=False))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
X = X.reshape(-1, 200, 7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(X_train, y_train, batch_size=4, epochs=11)
y_pred = model.predict_classes(X_test)


# In[ ]:
y_test_df = pd.DataFrame({0:y_test[:,0],1:y_test[:,1],2:y_test[:,2],3:y_test[:,3]})
y_test_df_idx=y_test_df.idxmax(axis=1)
print(confusion_matrix(y_test_df_idx, y_pred))
print(classification_report(y_test_df_idx, y_pred))
print('Accuracy %0.4f' % accuracy_score(y_test_df_idx, y_pred))

