#!/usr/bin/env python
# coding: utf-8

# In[455]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import jinja2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score


# In[456]:


df_train=pd.read_csv("Data.csv")


# In[457]:


df_train.columns


# In[458]:


df_train['Label'].values[0]


# In[459]:


df_train.head()


# In[460]:


df_train['Label'].unique()


# In[461]:


df_train['Sequential Internal Power'].fillna(df_train['Sequential Internal Power'].median(),inplace=True)


# In[462]:


df_train['Sequential Total Power'].fillna(df_train['Sequential Total Power'].median(),inplace=True)


# In[463]:


df_train.isnull().sum()


# In[464]:


df_train.drop(['Circuit'],axis=1,inplace=True)


# In[465]:


df_train=df_train.sample(frac=1).reset_index()


# In[466]:


df_train=df_train.drop(['index'],axis=1)


# In[467]:


df_train.head()


# In[468]:


df_train.shape


# In[469]:


df_train['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[470]:


x=df_train.drop(['Label'],axis=1)


# In[471]:


(df_train['Label']==0).sum()


# In[472]:


(df_train['Label']==1).sum()


# In[473]:


y=df_train['Label']


# In[474]:


df1=df_train[["Cell Internal Power (mW)","Net Switching Power (mW)","Total Dynamic Power (mW)","Cell Leakage Power (mW)","IO_Pad Internal Power","IO_Pad Switching Power","IO_Pad Leakage Power","IO_Pad Total Power","Memory Internal Power","Memory Switching Power","Memory Leakage Power","Memory Total Power","Black_Box Internal Power","Black_Box Switching Power","Black_Box Leakage Power","Black_Box Total Power","Clock_Network Internal Power","Clock_Network Switching Power","Clock_Network Leakage Power","Clock_Network Total Power","Register Internal Power","Register Switching Power","Register Leakage Power","Register Total Power","Sequential Internal Power","Sequential Switching Power","Sequential Leakage Power","Sequential Total Power","Combinational Internal Power","Combinational Switching Power","Combinational Leakage Power","Combinational Total Power","Total Internal Power","Total Switching Power","Total Leakage Power","Total Total Power"]]


# In[475]:


df2=df_train[["Number of ports","Number of nets","Number of cells","Number of combinational cells","Number of sequential cells","Number of macros/black boxes","Number of buf/inv","Number of references","Combinational area","Buf/Inv area","Noncombinational area","Macro/Black Box area","Total cell area"]]


# In[476]:


f, ax = plt.subplots(figsize=(12, 8))
corr = df_train.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Features Correlation', fontsize=18)


# In[477]:


corr


# In[ ]:





# In[478]:


import mpld3
mpld3.enable_notebook()


# In[479]:


col=['Number of ports', 'Number of nets', 'Number of cells',
       'Number of combinational cells', 'Number of sequential cells',
       'Number of macros/black boxes', 'Number of buf/inv',
       'Number of references', 'Combinational area', 'Buf/Inv area',
       'Noncombinational area', 'Macro/Black Box area', 'Total cell area',
       'Cell Internal Power (mW)', 'Net Switching Power (mW)',
       'Total Dynamic Power (mW)', 'Cell Leakage Power (mW)',
       'IO_Pad Internal Power', 'IO_Pad Switching Power',
       'IO_Pad Leakage Power', 'IO_Pad Total Power', 'Memory Internal Power',
       'Memory Switching Power', 'Memory Leakage Power', 'Memory Total Power',
       'Black_Box Internal Power', 'Black_Box Switching Power',
       'Black_Box Leakage Power', 'Black_Box Total Power',
       'Clock_Network Internal Power', 'Clock_Network Switching Power',
       'Clock_Network Leakage Power', 'Clock_Network Total Power',
       'Register Internal Power', 'Register Switching Power',
       'Register Leakage Power', 'Register Total Power',
       'Sequential Internal Power', 'Sequential Switching Power',
       'Sequential Leakage Power', 'Sequential Total Power',
       'Combinational Internal Power', 'Combinational Switching Power',
       'Combinational Leakage Power', 'Combinational Total Power',
       'Total Internal Power', 'Total Switching Power', 'Total Leakage Power',
       'Total Total Power']


# In[480]:


subset_df = df_train[col]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=col)
final_df = pd.concat([scaled_df, df_train['Label']], axis=1)
final_df.head()

# plot parallel coordinates
from pandas.plotting import parallel_coordinates
pc = parallel_coordinates(final_df, 'Label', color=('#FFE888', '#FF9999'))


# In[481]:


corr_matrix = x.corr()
threshold=0.96
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
x= x.drop(columns=drops)


# In[482]:


x.columns


# In[483]:


x.shape


# In[484]:


col=['Number of ports', 'Number of nets', 'Number of sequential cells',
       'Number of macros/black boxes', 'Number of references',
       'Macro/Black Box area', 'Cell Internal Power (mW)',
       'Net Switching Power (mW)', 'Cell Leakage Power (mW)',
       'IO_Pad Internal Power', 'IO_Pad Switching Power',
       'IO_Pad Leakage Power', 'IO_Pad Total Power', 'Memory Internal Power',
       'Memory Switching Power', 'Memory Leakage Power', 'Memory Total Power',
       'Black_Box Internal Power', 'Black_Box Switching Power',
       'Black_Box Leakage Power', 'Black_Box Total Power',
       'Clock_Network Internal Power', 'Clock_Network Switching Power',
       'Clock_Network Leakage Power', 'Sequential Internal Power',
       'Sequential Leakage Power']


# In[485]:


subset_df = df_train[col]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=col)
final_df = pd.concat([scaled_df, df_train['Label']], axis=1)
final_df.head()

# plot parallel coordinates
from pandas.plotting import parallel_coordinates
pc = parallel_coordinates(final_df, 'Label', color=('#FFE888', '#FF9999'))


# In[486]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


# In[487]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x_res, y_res = ros.fit_resample(x, y)


# In[488]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x_res,y_res,test_size=0.2,random_state=0)


# In[489]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


# In[490]:


model = Sequential()
model.add(Dense(16, input_shape=(x.shape[1],), activation='relu')) # Add an input shape! (features,)
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary() 


# In[491]:


model.compile(optimizer='Adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[492]:


es = EarlyStopping(monitor='val_accuracy', 
                                   mode='max', # don't minimize the accuracy!
                                   patience=10,
                                   restore_best_weights=True)


# In[493]:


history = model.fit(X_train
                    ,Y_train,
                    callbacks=[es],
                    epochs=100, # you can set this to a big number!
                    batch_size=10,
                    validation_split=0.2,
                    shuffle=True,
                    verbose=1)


# In[494]:


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


# In[495]:


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


# In[496]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# see how these are numbers between 0 and 1? 
model.predict(x) # prob of successes (survival)
np.round(model.predict(x),0) # 1 and 0 (survival or not)
y # 1 and 0 (survival or not)

# so we need to round to a whole number (0 or 1),
# or the confusion matrix won't work!
preds = np.round(model.predict(x),0)

# confusion matrix
print(confusion_matrix(y, preds)) # order matters! (actual, predicted)

## array([[490,  59],   ([[TN, FP],
##       [105, 235]])     [Fn, TP]])

print(classification_report(y, preds))


# In[497]:


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


# In[498]:


recall_score(Y_test,preds)


# In[499]:


x.columns


# In[500]:


x_res.shape


# In[501]:


y_res.shape


# In[502]:


y_res=pd.DataFrame(y_res)


# In[503]:


(y_res['Label']==1).sum()


# In[ ]:





# In[504]:


type(x)


# In[505]:


type(y)


# In[ ]:





# In[506]:


x_res.shape


# In[507]:


y_res.shape


# In[508]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42)


# In[509]:


model=RandomForestClassifier()


# In[510]:


model.fit(x_train,y_train)


# In[511]:


x_train.shape


# In[512]:


y_train.shape


# In[513]:


x_test.shape


# In[514]:


x_train


# In[515]:


score=model.score(x_train,y_train)


# In[516]:


score


# In[517]:


score=model.score(x_test,y_test)


# In[518]:


score


# In[519]:


y_pred=model.predict(x_test)


# In[520]:


confusion_matrix(y_test,y_pred)


# In[521]:


param_grid = {
    'n_estimators': [100],
    'max_features': ['sqrt'],
    'max_depth': [20],
    'max_leaf_nodes': [25]
}


# In[522]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# In[523]:


grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid)
grid_search.fit(x_train, y_train)
print(grid_search.best_estimator_)


# In[524]:


model_grid = RandomForestClassifier(max_depth=20,
                                    max_features='sqrt',
                                    max_leaf_nodes=30,
                                    n_estimators=150)
model_grid.fit(x_train, y_train)
y_pred_grid = model_grid.predict(x_test)
print(classification_report(y_pred_grid, y_test))


# In[525]:


score=model_grid.score(x_test,y_test)


# In[526]:


score


# In[527]:


confusion_matrix(y_test,y_pred_grid)


# In[528]:


from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):

      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }


# In[529]:


def plot_result(x_label, y_label, plot_title, train_data, val_data):
      
    
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()


# In[530]:


result = cross_validation(model_grid, x_res, y_res, 5)
print(result)


# In[531]:


model_name = "Random Forest"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            result["Training Accuracy scores"],
            result["Validation Accuracy scores"])




# In[532]:


#random_search = RandomizedSearchCV(RandomForestClassifier(),
#                                   param_grid)
#random_search.fit(x_train, y_train)
#print(random_search.best_estimator_)


# In[533]:


model_random = RandomForestClassifier(max_depth=12,
                                      max_features='sqrt',
                                      max_leaf_nodes=25,
                                      n_estimators=150)
model_random.fit(x_train, y_train)
y_pred_rand = model.predict(x_test)
print(classification_report(y_pred_rand, y_test))


# In[534]:


score=model_random.score(x_test,y_test)


# In[535]:


score


# In[536]:


confusion_matrix(y_test,y_pred_rand)


# In[537]:


from xgboost import XGBClassifier
modelXGBOOSTER = XGBClassifier()
modelXGBOOSTER.fit(x_train, y_train)


# In[548]:


score=modelXGBOOSTER.score(x_test,y_test)


# In[ ]:





# In[ ]:





# In[549]:


score


# In[550]:


y_pred = modelXGBOOSTER.predict(x_test)


# In[552]:


confusion_matrix(y_test,y_pred)


# In[553]:


from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):

      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }


# In[554]:


def plot_result(x_label, y_label, plot_title, train_data, val_data):
      
    
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()


# In[555]:


result = cross_validation(modelXGBOOSTER, x_res, y_res, 5)
print(result)


# In[556]:


model_name = "XGBOOSTER"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            result["Training Accuracy scores"],
            result["Validation Accuracy scores"])


# In[557]:


# check xgboost version
import xgboost as xgb


# In[ ]:





# In[558]:


ensemblemodel =xgb.XGBRFClassifier()


# In[ ]:





# In[559]:


ensemblemodel = xgb.XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.8)


# In[560]:


ensemblemodel.fit(x_train, y_train)


# In[561]:


score=ensemblemodel.score(x_test,y_test)


# In[562]:


score


# In[563]:


df1_train=pd.read_csv("data_1.csv")


# In[564]:


df2_train=pd.read_csv("data_2.csv")


# In[565]:


df3_train=pd.read_csv("data_3.csv")


# In[566]:


df4_train=pd.read_csv("data_4.csv")


# In[567]:


df5_train=pd.read_csv("data_5.csv")


# In[568]:


df5_train.head


# In[569]:


df5_train.columns


# In[570]:


df1_train.isnull().sum()


# In[571]:


df2_train.isnull().sum()


# In[572]:


df2_train['Sequential Internal Power'].fillna(df_train['Sequential Internal Power'].median(),inplace=True)


# In[573]:


df2_train['Sequential Total Power'].fillna(df_train['Sequential Total Power'].median(),inplace=True)


# In[574]:


df3_train.isnull().sum()


# In[575]:


df4_train.isnull().sum()


# In[576]:


df1_train.isnull().sum()


# In[577]:


x1=df1_train[['Number of ports', 'Number of nets', 'Number of sequential cells',
       'Number of macros/black boxes', 'Number of references',
       'Macro/Black Box area', 'Cell Internal Power (mW)',
       'Net Switching Power (mW)', 'Cell Leakage Power (mW)',
       'IO_Pad Internal Power', 'IO_Pad Switching Power',
       'IO_Pad Leakage Power', 'IO_Pad Total Power', 'Memory Internal Power',
       'Memory Switching Power', 'Memory Leakage Power', 'Memory Total Power',
       'Black_Box Internal Power', 'Black_Box Switching Power',
       'Black_Box Leakage Power', 'Black_Box Total Power',
       'Clock_Network Internal Power', 'Clock_Network Switching Power',
       'Clock_Network Leakage Power', 'Sequential Internal Power',
       'Sequential Leakage Power']]


# In[578]:


df1_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[579]:


df1_train['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[580]:


y1=df1_train['Label']


# In[581]:


y1.shape


# In[582]:


y1.shape


# In[583]:


score=modelXGBOOSTER.score(x1,y1)

score
# In[584]:


print(score)


# In[585]:


y_pred = modelXGBOOSTER.predict(x1)


# In[586]:


confusion_matrix(y1,y_pred)


# In[587]:


x2=df2_train[['Number of ports', 'Number of nets', 'Number of sequential cells',
       'Number of macros/black boxes', 'Number of references',
       'Macro/Black Box area', 'Cell Internal Power (mW)',
       'Net Switching Power (mW)', 'Cell Leakage Power (mW)',
       'IO_Pad Internal Power', 'IO_Pad Switching Power',
       'IO_Pad Leakage Power', 'IO_Pad Total Power', 'Memory Internal Power',
       'Memory Switching Power', 'Memory Leakage Power', 'Memory Total Power',
       'Black_Box Internal Power', 'Black_Box Switching Power',
       'Black_Box Leakage Power', 'Black_Box Total Power',
       'Clock_Network Internal Power', 'Clock_Network Switching Power',
       'Clock_Network Leakage Power', 'Sequential Internal Power',
       'Sequential Leakage Power']]


# In[588]:


df2_train.shape


# In[589]:


x2.shape


# In[ ]:





# In[590]:


df2_train['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[591]:


y2=df2_train['Label']


# In[592]:


x2=df2_train[['Number of ports', 'Number of nets', 'Number of sequential cells',
       'Number of macros/black boxes', 'Number of references',
       'Macro/Black Box area', 'Cell Internal Power (mW)',
       'Net Switching Power (mW)', 'Cell Leakage Power (mW)',
       'IO_Pad Internal Power', 'IO_Pad Switching Power',
       'IO_Pad Leakage Power', 'IO_Pad Total Power', 'Memory Internal Power',
       'Memory Switching Power', 'Memory Leakage Power', 'Memory Total Power',
       'Black_Box Internal Power', 'Black_Box Switching Power',
       'Black_Box Leakage Power', 'Black_Box Total Power',
       'Clock_Network Internal Power', 'Clock_Network Switching Power',
       'Clock_Network Leakage Power', 'Sequential Internal Power',
       'Sequential Leakage Power']]


# In[593]:


y2.shape


# In[594]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x2_res, y2_res = ros.fit_resample(x2, y2)


# In[595]:


x2_res.shape


# In[596]:


y2_res.shape


# In[597]:


score=modelXGBOOSTER.score(x2_res,y2_res)


# In[ ]:





# In[598]:


score


# In[599]:


print(score)


# In[600]:


y_pred = modelXGBOOSTER.predict(x2_res)


# In[601]:


confusion_matrix(y2_res,y_pred)


# In[602]:


x3=df3_train[['Number of ports', 'Number of nets', 'Number of sequential cells',
       'Number of macros/black boxes', 'Number of references',
       'Macro/Black Box area', 'Cell Internal Power (mW)',
       'Net Switching Power (mW)', 'Cell Leakage Power (mW)',
       'IO_Pad Internal Power', 'IO_Pad Switching Power',
       'IO_Pad Leakage Power', 'IO_Pad Total Power', 'Memory Internal Power',
       'Memory Switching Power', 'Memory Leakage Power', 'Memory Total Power',
       'Black_Box Internal Power', 'Black_Box Switching Power',
       'Black_Box Leakage Power', 'Black_Box Total Power',
       'Clock_Network Internal Power', 'Clock_Network Switching Power',
       'Clock_Network Leakage Power', 'Sequential Internal Power',
       'Sequential Leakage Power']]


# In[603]:


df3_train.shape


# In[604]:


df3_train['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[605]:


y3=df3_train['Label']


# In[606]:


y3.shape


# In[607]:


score=modelXGBOOSTER.score(x3,y3)


# In[608]:


score


# In[609]:


print(score)


# In[610]:


y_pred = modelXGBOOSTER.predict(x3)


# In[611]:


confusion_matrix(y3,y_pred)


# In[612]:


x4=df4_train[['Number of ports', 'Number of nets', 'Number of sequential cells',
       'Number of macros/black boxes', 'Number of references',
       'Macro/Black Box area', 'Cell Internal Power (mW)',
       'Net Switching Power (mW)', 'Cell Leakage Power (mW)',
       'IO_Pad Internal Power', 'IO_Pad Switching Power',
       'IO_Pad Leakage Power', 'IO_Pad Total Power', 'Memory Internal Power',
       'Memory Switching Power', 'Memory Leakage Power', 'Memory Total Power',
       'Black_Box Internal Power', 'Black_Box Switching Power',
       'Black_Box Leakage Power', 'Black_Box Total Power',
       'Clock_Network Internal Power', 'Clock_Network Switching Power',
       'Clock_Network Leakage Power', 'Sequential Internal Power',
       'Sequential Leakage Power']]


# In[613]:


df4_train.shape


# In[614]:


df4_train['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[615]:


y4=df4_train['Label']


# In[616]:


y4.shape


# In[617]:


score=modelXGBOOSTER.score(x4,y4)


# In[618]:


score


# In[619]:


print(score)


# In[620]:


y_pred = modelXGBOOSTER.predict(x4)


# In[621]:


confusion_matrix(y4,y_pred)


# In[622]:


x5=df5_train[['Number of ports', 'Number of nets', 'Number of sequential cells',
       'Number of macros/black boxes', 'Number of references',
       'Macro/Black Box area', 'Cell Internal Power (mW)',
       'Net Switching Power (mW)', 'Cell Leakage Power (mW)',
       'IO_Pad Internal Power', 'IO_Pad Switching Power',
       'IO_Pad Leakage Power', 'IO_Pad Total Power', 'Memory Internal Power',
       'Memory Switching Power', 'Memory Leakage Power', 'Memory Total Power',
       'Black_Box Internal Power', 'Black_Box Switching Power',
       'Black_Box Leakage Power', 'Black_Box Total Power',
       'Clock_Network Internal Power', 'Clock_Network Switching Power',
       'Clock_Network Leakage Power', 'Sequential Internal Power',
       'Sequential Leakage Power']]


# In[623]:


df5_train.shape


# In[624]:


df5_train['Label'].replace({"'Trojan Free'":0,"'Trojan Infected'":1},inplace=True)


# In[625]:


y5=df5_train['Label']


# In[626]:


y5.shape


# In[627]:


score=modelXGBOOSTER.score(x5,y5)


# In[628]:


score


# In[629]:


print(score)


# In[630]:


y_pred = modelXGBOOSTER.predict(x5)


# In[631]:


confusion_matrix(y5,y_pred)


# In[ ]:





# In[633]:


pip install Pypeteer


# In[ ]:




