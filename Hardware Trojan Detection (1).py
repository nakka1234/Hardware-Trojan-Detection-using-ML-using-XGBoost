#!/usr/bin/env python
# coding: utf-8

# In[266]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jinja2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score


# In[267]:


df_train=pd.read_csv("Data.csv")


# In[268]:


df_train.columns


# In[269]:


df_train['Label'].values[0]


# In[270]:


df_train['Label'].unique()


# In[271]:


len(df_train.columns)


# In[272]:


df_train.Label.value_counts()["'Trojan Free'"]


# In[273]:


df_train.Label.value_counts()["'Trojan Infected'"]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[274]:


df_train.head()


# In[275]:


df_train['Circuit'].unique


# In[276]:


(df_train['Label']=="'Trojan Free'").sum()


# In[277]:


(df_train['Label']=="'Trojan Infected'").sum()


# In[278]:


df_train.isnull().sum()


# In[279]:


df_train['Sequential Internal Power'].fillna(df_train['Sequential Internal Power'].median(),inplace=True)


# In[280]:


df_train['Sequential Total Power'].fillna(df_train['Sequential Total Power'].median(),inplace=True)


# In[281]:


df_train.isnull().sum()


# In[282]:


df_train.drop(['Circuit'],axis=1,inplace=True)


# In[283]:


df_train.shape


# In[284]:


df_train['Label'].unique()


# In[285]:


df_train['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[286]:


df_train['Label'].unique()


# In[287]:


df1=df_train[["Number of ports","Number of nets","Number of cells","Number of combinational cells","Number of sequential cells","Number of macros/black boxes","Number of buf/inv","Number of references","Combinational area","Buf/Inv area","Noncombinational area","Macro/Black Box area","Total cell area","Label" ]]


# In[288]:


df1.shape


# In[289]:


x=df1.drop(['Label'],axis=1)


# In[290]:




y=df1['Label']


# In[291]:


x.shape


# In[292]:


df1.corr()


# In[293]:


corr_matrix = x.corr()
threshold=0.9
iters = range(len(corr_matrix.columns) - 1)
drop_cols = []

    # Iterate through the correlation matrix and compare correlations
for i in iters:
    for j in range(i+1):
        item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
        col = item.columns
        row = item.index
        val = abs(item.values)

            # If correlation exceeds the threshold
        if val >= threshold:
                # Print the correlated features and the correlation value
                #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
            drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
drops = set(drop_cols)
df1 = df1.drop(columns=drops)


# In[294]:


df1


# In[295]:


df1.shape


# In[296]:


df2=df_train[["Cell Internal Power (mW)","Net Switching Power (mW)","Total Dynamic Power (mW)","Cell Leakage Power (mW)","IO_Pad Internal Power","IO_Pad Switching Power","IO_Pad Leakage Power","IO_Pad Total Power","Memory Internal Power","Memory Switching Power","Memory Leakage Power","Memory Total Power","Black_Box Internal Power","Black_Box Switching Power","Black_Box Leakage Power","Black_Box Total Power","Clock_Network Internal Power","Clock_Network Switching Power","Clock_Network Leakage Power","Clock_Network Total Power","Register Internal Power","Register Switching Power","Register Leakage Power","Register Total Power","Sequential Internal Power","Sequential Switching Power","Sequential Leakage Power","Sequential Total Power","Combinational Internal Power","Combinational Switching Power","Combinational Leakage Power","Combinational Total Power","Total Internal Power","Total Switching Power","Total Leakage Power","Total Total Power","Label"]]


# In[297]:


df2.shape


# In[298]:


corr_matrix = df2.corr()
threshold=0.95
iters = range(len(corr_matrix.columns) - 1)
drop_cols = []

    # Iterate through the correlation matrix and compare correlations
for i in iters:
    for j in range(i+1):
        item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
        col = item.columns
        row = item.index
        val = abs(item.values)

            # If correlation exceeds the threshold
        if val >= threshold:
                # Print the correlated features and the correlation value
                #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
            drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
drops = set(drop_cols)
df3 = df2.drop(columns=drops)


# ## 

# In[299]:


df3.shape


# In[300]:


df1.shape


# In[301]:


df1.shape


# df2.shape

# In[302]:


df3.shape


# In[303]:


df4 = pd.concat([df1, df3], axis=1, join='inner')


# In[304]:


df4


# In[ ]:





# In[305]:


df4.shape


# In[306]:


model=RandomForestClassifier()


# In[ ]:





# In[ ]:





# In[307]:


y


# In[308]:


df4


# In[309]:


df4=df4.drop(['Label'],axis=1)


# In[310]:


df4.columns


# In[311]:


model=RandomForestClassifier()


# In[312]:


df4.columns


# In[313]:


X_train,X_test,Y_train,Y_test=train_test_split(df4,y,test_size=0.2,random_state=42)


# In[314]:


model.fit(X_train,Y_train)


# In[ ]:





# In[ ]:





# In[315]:


score=model.score(X_test,Y_test)


# In[316]:


score


# In[317]:


y_pred=model.predict(X_test)


# In[ ]:





# In[318]:


confusion_matrix(Y_test,y_pred)


# In[319]:


df1_train=pd.read_csv("data_1.csv")


# In[320]:


df1_train.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[321]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[322]:


model=RandomForestClassifier()


# In[323]:


model.fit(X_train,Y_train)


# In[324]:


score=model.score(X_test,Y_test)


# In[325]:


score


# In[326]:


df1=df_train[["Cell Internal Power (mW)","Net Switching Power (mW)","Total Dynamic Power (mW)","Cell Leakage Power (mW)","IO_Pad Internal Power","IO_Pad Switching Power","IO_Pad Leakage Power","IO_Pad Total Power","Memory Internal Power","Memory Switching Power","Memory Leakage Power","Memory Total Power","Black_Box Internal Power","Black_Box Switching Power","Black_Box Leakage Power","Black_Box Total Power","Clock_Network Internal Power","Clock_Network Switching Power","Clock_Network Leakage Power","Clock_Network Total Power","Register Internal Power","Register Switching Power","Register Leakage Power","Register Total Power","Sequential Internal Power","Sequential Switching Power","Sequential Leakage Power","Sequential Total Power","Combinational Internal Power","Combinational Switching Power","Combinational Leakage Power","Combinational Total Power","Total Internal Power","Total Switching Power","Total Leakage Power","Total Total Power","Label"]]


# In[327]:


df1.shape


# In[328]:


x=df1.drop(['Label'],axis=1)


# In[329]:


y=df1['Label']


# In[330]:


x.shape


# In[331]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[332]:


model=RandomForestClassifier()


# In[333]:


model.fit(X_train,Y_train)


# In[334]:


score=model.score(X_test,Y_test)


# In[335]:


score


# In[ ]:





# In[ ]:





# In[336]:


df_train.corr()


# In[337]:


cor_matrix = df_train.corr().abs() 
print(cor_matrix)


# In[338]:


upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool)) 
print(upper_tri)


# In[339]:


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)] 
print()


# In[340]:


df1_train=df_train


# In[341]:


df1_train.drop(['Number of cells', 'Number of combinational cells', 'Number of buf/inv', 'Combinational area', 'Buf/Inv area', 'Noncombinational area'], axis=1)


# In[342]:


df2_train=df1_train.drop(['Total cell area', 'Total Dynamic Power (mW)', 'Clock_Network Total Power', 'Register Internal Power', 'Register Switching Power', 'Register Leakage Power'],axis=1)


# In[343]:


df2_train.shape


# In[344]:


df1_train=df2_train.drop(['Register Total Power', 'Sequential Switching Power', 'Sequential Total Power', 'Combinational Internal Power', 'Combinational Switching Power', 'Combinational Leakage Power'],axis=1)


# In[345]:


df1_train.shape


# In[346]:


df2_train=df1_train.drop([ 'Combinational Total Power', 'Total Internal Power', 'Total Switching Power', 'Total Leakage Power', 'Total Total Power'],axis=1)


# In[347]:


df2_train.shape


# In[348]:


print(to_drop)


# In[349]:


df2_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[350]:


df_train.shape


# In[351]:


df1_train.columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[352]:


print(to_drop)


# In[353]:


len(to_drop)


# In[ ]:





# In[ ]:





# In[ ]:





# In[354]:


## train_test_split


# In[355]:


## features


# In[ ]:





# In[356]:


X=df_train.drop(['Label'],axis=1) 


# In[357]:


## label


# In[358]:


y=df_train['Label']


# In[359]:


X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[360]:


X_train.shape


# In[361]:


X_test.shape


# In[362]:


Y_train.shape


# In[363]:


Y_test.shape


# In[364]:


##  balancing the dataset unsing smotek


# In[365]:


df1_train=df2_train.drop(['Label'],axis=1)


# In[366]:


df1_train.shape


# In[367]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0) 

kmeans.fit(df1_train)


# In[368]:


kmeans.cluster_centers_


# In[369]:


kmeans.inertia_


# In[370]:


labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))


# In[371]:


df1_train.shape


# In[372]:


X1_train,X1_test,Y1_train,Y1_test=train_test_split(df1_train,y,test_size=0.2,random_state=42)


# In[373]:


## standarization(optional)


# In[374]:


## Model


# In[375]:


model=RandomForestClassifier()


# In[376]:


model.fit(X_train,Y_train)


# In[377]:


score=model.score(X_test,Y_test)


# In[378]:


score


# In[379]:


model2=RandomForestClassifier()


# In[380]:


model2.fit(X1_train,Y1_train)


# In[381]:


X1_train.shape


# In[382]:


score=model2.score(X1_test,Y1_test)


# In[383]:


print(score)


# In[384]:


y_pred=model2.predict(X1_test)


# In[385]:


confusion_matrix(Y1_test,y_pred)


# In[386]:


X1_test.shape


# In[387]:


y_pred=model.predict(X_test)


# In[388]:


confusion_matrix(Y_test,y_pred)


# In[389]:


precision_score(Y_test,y_pred)


# In[390]:


#pip install imblearn


# In[391]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)


# In[392]:


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(Y_train_res.shape))
  
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_res == 0)))


# In[393]:


model=RandomForestClassifier()


# In[394]:


model.fit(X_train_res,Y_train_res)


# In[395]:


score=model.score(X_test,Y_test)


# In[396]:


score


# In[397]:


y_pred=model.predict(X_test)


# In[398]:


confusion_matrix(Y_test,y_pred)


# In[399]:


#!pip install tensorflow


# In[400]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


# In[401]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


# In[402]:


model = Sequential()
model.add(Dense(16, input_shape=(X.shape[1],), activation='relu')) # Add an input shape! (features,)
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary() 


# In[403]:


model.compile(optimizer='Adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:





# In[404]:


es = EarlyStopping(monitor='val_accuracy', 
                                   mode='max', # don't minimize the accuracy!
                                   patience=10,
                                   restore_best_weights=True)


# In[405]:


history = model.fit(X_train_res,Y_train_res,callbacks=[es], epochs=80,batch_size=10,validation_split=0.2,shuffle=True,verbose=1)


# In[406]:


history_dict = history.history
# Learning curve(Loss)
# let's see the training and validation loss by epoch

# loss
loss_values = history_dict['loss'] # you can change this
val_loss_values = history_dict['val_loss'] # you can also change this

# range of X (no. of epochs)
epochs = range(1, len(loss_values) + 1) 

# plot
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[407]:


# Learning curve(accuracy)
# let's see the training and validation accuracy by epoch

# accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# orange is for "orange"
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# this is the max value - should correspond to
# the HIGHEST train accuracy
np.max(val_acc)


# In[408]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# see how these are numbers between 0 and 1? 
model.predict(X) # prob of successes (survival)
np.round(model.predict(X),0) # 1 and 0 (survival or not)
y # 1 and 0 (survival or not)

# so we need to round to a whole number (0 or 1),
# or the confusion matrix won't work!
preds = np.round(model.predict(X),0)

# confusion matrix
print(confusion_matrix(y, preds)) # order matters! (actual, predicted)

## array([[490,  59],   ([[TN, FP],
##       [105, 235]])     [Fn, TP]])

print(classification_report(y, preds))
##               precision    recall  f1-score   support
## 
##            0       0.82      0.89      0.86       549
##            1       0.80      0.69      0.74       340
## 
##     accuracy                           0.82       889
##    macro avg       0.81      0.79      0.80       889
## weighted avg       0.81      0.82      0.81       889


# In[409]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# see how these are numbers between 0 and 1? 
model.predict(X_test) # prob of successes (survival)
np.round(model.predict(X_test),0) # 1 and 0 (survival or not)
Y_test # 1 and 0 (survival or not)

# so we need to round to a whole number (0 or 1),
# or the confusion matrix won't work!
preds = np.round(model.predict(X_test),0)

# confusion matrix
print(confusion_matrix(Y_test, preds)) # order matters! (actual, predicted)

## array([[490,  59],   ([[TN, FP],
##       [105, 235]])     [Fn, TP]])

print(classification_report(Y_test, preds))
##               precision    recall  f1-score   support
## 
##            0       0.82      0.89      0.86       549
##            1       0.80      0.69      0.74       340
## 
##     accuracy                           0.82       889
##    macro avg       0.81      0.79      0.80       889
## weighted avg       0.81      0.82      0.81       889


# In[ ]:





# In[ ]:





# In[410]:


recall_score(Y_test,y_pred)


# In[411]:


d1_frame=pd.read_csv("data_1.csv")


# d1_train.drop(['Circuit'],axis=1,inplace=True)

# In[412]:


d1_frame.drop(['Circuit'],axis=1,inplace=True)


# In[413]:


d1_frame['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[414]:


(d1_frame['Label']==0).sum()


# In[415]:


(d1_frame['Label']==1).sum()


# In[416]:


x1=d1_frame.drop(['Label'],axis=1) 


# In[417]:


y1=d1_frame['Label'];


# In[418]:


score=model.score(x1,y1)


# In[ ]:


print(score)


# In[ ]:


y1_pred=model.predict(x1)


# In[ ]:


confusion_matrix(y1,y1_pred)


# In[ ]:


d2_frame=pd.read_csv("data_2.csv")


# In[ ]:


d2_frame.drop(['Circuit'],axis=1,inplace=True)


# In[ ]:


d2_frame['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[ ]:


d2_frame.isnull().sum()


# In[ ]:


d2_frame.shape


# In[ ]:


(d2_frame['Label']==0).sum()


# In[ ]:


(d2_frame['Label']==1).sum()


# In[ ]:


d2_frame['Sequential Internal Power'].fillna(d2_frame['Sequential Internal Power'].median(),inplace=True)


# In[ ]:





# In[ ]:



d2_frame['Sequential Total Power'].fillna(d2_frame['Sequential Total Power'].median(),inplace=True)


# In[ ]:


x2=d2_frame.drop(['Label'],axis=1)


# In[ ]:


y2=d2_frame['Label']


# In[ ]:


score=model.score(x2,y2)


# In[ ]:


print(score)


# In[ ]:





# In[ ]:


confusion_matrix(y2,y2_pred)


# In[ ]:


score=model.score(x2,y2)


# In[ ]:


print(score)


# In[ ]:


y2_pred=model.predict(x2)


# In[ ]:


y2_pred=model.predict(x2)


# In[ ]:


d3_frame=pd.read_csv("data_3.csv")


# In[ ]:


d3_frame.drop(['Circuit'],axis=1,inplace=True)


# In[ ]:


d3_frame['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[ ]:


x3=d3_frame.drop(['Label'],axis=1)


# In[ ]:


y3=d3_frame['Label'];


# In[ ]:


score=model.score(x3,y3)


# In[ ]:


print(score)


# In[ ]:


y3_pred=model.predict(x3)


# In[ ]:


confusion_matrix(y3,y3_pred)


# In[ ]:


score=model.score(x3,y3)


# In[ ]:


print(score)


# In[ ]:


y3_pred=model.predict(x3)


# In[ ]:


y3_pred=model.predict(x3)


# In[ ]:


d4_frame=pd.read_csv("data_4.csv")


# In[ ]:


d4_frame.drop(['Circuit'],axis=1,inplace=True)


# In[ ]:


d4_frame['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[ ]:


(d4_frame['Label']==0).sum()


# In[ ]:


(d4_frame['Label']==1).sum()


# In[ ]:


x4=d4_frame.drop(['Label'],axis=1)


# In[ ]:


y4=d4_frame['Label'];


# In[ ]:


score=model.score(x4,y4)


# In[ ]:


print(score)


# In[ ]:


y4_pred=model.predict(x4)


# In[ ]:


confusion_matrix(y4,y4_pred)


# In[ ]:


score=model.score(x3,y3)


# In[ ]:


print(score)


# In[ ]:


y4_pred=model.predict(x4)


# In[ ]:


y4_pred=model.predict(x4)


# In[ ]:


df_train['Number of macros/black boxes'].unique()


# In[ ]:





# In[ ]:


## SMOTEK


# In[ ]:


df_train.columns


# In[ ]:


df_train.shape


# In[ ]:


df4


# In[ ]:


y=df_train['Label']


# In[ ]:


y


# In[ ]:


df_train.isnull().sum()


# In[ ]:


x=df4


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x,y)


# In[ ]:


x_res


# In[ ]:


(y_res==0).sum()


# x1=df_train[df_train["Label"]=="'Trojan Free'"]

# (y_res==1).sum()

# In[ ]:


(y_res==1).sum()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42)


# In[ ]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[419]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


# In[420]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


# In[421]:


model = Sequential()
model.add(Dense(16, input_shape=(x.shape[1],), activation='relu')) # Add an input shape! (features,)
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary() 


# In[422]:


model.compile(optimizer='Adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[423]:


es = EarlyStopping(monitor='val_accuracy', 
                                   mode='max', # don't minimize the accuracy!
                                   patience=10,
                                   restore_best_weights=True)


# In[424]:


history = model.fit(x_train,y_train,callbacks=[es],epochs=80,batch_size=10,validation_split=0.2,shuffle=True,verbose=1)
    


# In[ ]:


history_dict = history.history
# Learning curve(Loss)
# let's see the training and validation loss by epoch

# loss
loss_values = history_dict['loss'] # you can change this
val_loss_values = history_dict['val_loss'] # you can also change this

# range of X (no. of epochs)
epochs = range(1, len(loss_values) + 1) 

# plot
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# Learning curve(accuracy)
# let's see the training and validation accuracy by epoch

# accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# orange is for "orange"
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# this is the max value - should correspond to
# the HIGHEST train accuracy
np.max(val_acc)


# In[425]:


x


# In[426]:


y


# In[427]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)


# In[428]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42)


# In[429]:


from sklearn.neural_network import MLPClassifier


classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)


# In[430]:


classifier.fit(x_train, y_train)


# In[431]:


y_pred = classifier.predict(x_test)


# In[432]:


cm = confusion_matrix(y_pred, y_test)


# In[433]:


cm


# In[ ]:





# In[434]:


score=classifier.score(x_test,y_test)


# In[435]:


score


# In[436]:


parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}


# In[437]:


from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=5)
clf.fit(x_train, y_train)


# In[438]:


clf.best_params_


# In[439]:


y_pred = clf.predict(x_test)


# In[ ]:





# In[440]:


cm = confusion_matrix(y_pred, y_test)


# In[441]:


cm


# In[442]:


score=clf.score(x_test,y_test)


# In[443]:


score


# In[444]:


from sklearn.svm import LinearSVC
lin_clf = LinearSVC(random_state=42)
lin_clf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = lin_clf.predict(x_train)
accuracy_score(y_train, y_pred)


# In[445]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
x_test_scaled = scaler.transform(x_test.astype(np.float32))
lin_clf = LinearSVC(random_state=42)
lin_clf.fit(x_train_scaled, y_train)
y_pred = lin_clf.predict(x_train_scaled)
accuracy_score(y_train, y_pred)


# In[446]:


from sklearn.svm import SVC
svm_clf = SVC(gamma="scale")
svm_clf.fit(x_train, y_train) # We use an SVC with an RBF kernel
y_pred = svm_clf.predict(x_train_scaled)
accuracy_score(y_train, y_pred)


# In[447]:


lin_clf = LinearSVC(random_state=42)


# In[448]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform


# In[449]:


param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3)
rnd_search_cv.fit(x_train_scaled, y_train)


# In[450]:


rnd_search_cv.best_estimator_


# In[451]:


rnd_search_cv.best_score_


# In[452]:


rnd_search_cv.best_estimator_.fit(x_train, y_train)


# In[453]:


from sklearn.metrics import mean_squared_error
y_pred = rnd_search_cv.best_estimator_.predict(x_train)
mse = mean_squared_error(y_train, y_pred)
np.sqrt(mse)


# In[454]:


y_pred = rnd_search_cv.best_estimator_.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
np.sqrt(mse)


# In[455]:


y_pred = rnd_search_cv.best_estimator_.predict(x_test)
accuracy_score(y_test, y_pred)


# In[456]:


cm = confusion_matrix(y_test, y_pred)


# In[457]:


cm


# In[562]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)


# In[563]:


y_pred = classifier.predict(x_test)


# In[564]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)


# In[565]:


cm


# In[566]:


print(ac)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




