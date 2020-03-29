#Archel Aguilar 
#This is a test of classification regression on credit cards
#
# Data description
#• ID: ID of each client
#• LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
#• SEX: Gender (1=male, 2=female)
#• EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
#• MARRIAGE: Marital status (1=married, 2=single, 3=others)
#• AGE: Age in years
#• PAY_PC1, PAY_PC2, PAY_PC3: First three Principal Components of repayment status from April to
#September, 2005
#• AMT_PC1, AMT_PC2, AMT_PC3, AMT_PC4, AMT_PC5, AMT_PC6, AMT_PC7: First seven Principal
#Components of the bill statement amount and the amount of previous payments from April to September,
#2005
#• default: Default payment next month (1=yes, 0=no)


#load libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print(pd.__version__)

os.chdir('C:/Users/arche/Documents/UTS/Python-References/94691 Deep Learning/')

print(os.getcwd())

cdf = pd.read_csv('Credit card\CreditPython\AT3_credit_train_STUDENT.csv')

print(type(cdf))

print(cdf.shape)

print(cdf.describe())

print(cdf.info())

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 23101 entries, 0 to 23100
# Data columns (total 17 columns):
#  #   Column     Non-Null Count  Dtype  
# ---  ------     --------------  -----  
#  0   ID         23101 non-null  int64  
#  1   LIMIT_BAL  23101 non-null  float64
#  2   SEX        23101 non-null  object 
#  3   EDUCATION  23101 non-null  int64  
#  4   MARRIAGE   23101 non-null  int64  
#  5   AGE        23101 non-null  int64  
#  6   PAY_PC1    23101 non-null  float64
#  7   PAY_PC2    23101 non-null  float64
#  8   PAY_PC3    23101 non-null  float64
#  9   AMT_PC1    23101 non-null  float64
#  10  AMT_PC2    23101 non-null  float64
#  11  AMT_PC3    23101 non-null  float64
#  12  AMT_PC4    23101 non-null  float64
#  13  AMT_PC5    23101 non-null  float64
#  14  AMT_PC6    23101 non-null  float64
#  15  AMT_PC7    23101 non-null  float64
#  16  default    23101 non-null  object 
# dtypes: float64(11), int64(4), object(2)

#looks like there is no NaNs but we need to analyse further. 
#use profiler

#import pandas_profiling

#profile = pandas_profiling.ProfileReport(cdf)
#profile.to_file(output_file='CreditCardProfile.html')

#pandas_profiling.ProfileReport(cdf)

#----------SEX
print(cdf.SEX.describe())

plt.hist(cdf.SEX)
#there is some odd values for sex eg. dolphin, cat, dog

cdf.SEX.value_counts()
# 2          13854
# 1           9244
# cat            1
# dolphin        1
# dog            1
# Name: SEX, dtype: int64

cdf.SEX.replace('cat', np.nan, inplace=True)
cdf.SEX.replace('dolphin', np.nan, inplace=True)
cdf.SEX.replace('dog', np.nan, inplace=True)

# only 3 records so these could be dropped. 
cdf.shape
cdf = cdf.dropna()
cdf.shape

#replace SEX with text
cdf.SEX.replace({'2':'Female', '1':'Male'}, inplace=True)

#-------LIMIT BAL
print(cdf.LIMIT_BAL.describe())
plt.hist(cdf.LIMIT_BAL)
#there are -99 values

cdf.LIMIT_BAL[cdf.LIMIT_BAL == -99].count()
#50 records wth -99 limit balance
#50/23098 = 0.002 is a low % overall and could be dropped. 

cdf.LIMIT_BAL.replace(-99, np.nan, inplace=True)
cdf.LIMIT_BAL.isnull().sum()

#-------EDUCATION
print(cdf.EDUCATION.describe())
plt.hist(cdf.EDUCATION)
type(cdf.EDUCATION.values)

cdf.EDUCATION.value_counts()
# 2    10722
# 1     8192
# 3     3820
# 5      229
# 4       88
# 6       36
# 0       11

#merge 5,6 into others (4) category
cdf.EDUCATION.replace({5:4, 6:4}, inplace=True)
   
#it also needs to be converted to an object
cdf.EDUCATION = cdf.EDUCATION.astype(str)
cdf.EDUCATION.describe()

#replace education with text
cdf.EDUCATION.replace({'1':'Grad', '2':'Uni', '3':'HighSchool', '4':'Other'}, inplace=True)
cdf.EDUCATION.replace({'0':'Other'}, inplace=True)

#----------MARRIAGE
print(cdf.MARRIAGE.describe())
plt.hist(cdf.MARRIAGE)
cdf.MARRIAGE.value_counts()

# 2    12304
# 1    10507
# 3      249
# 0       38

#there are 38 with 0. Move them to others
cdf.MARRIAGE.replace({0:3}, inplace=True)
cdf.MARRIAGE = cdf.MARRIAGE.astype(str)
cdf.MARRIAGE.describe()

#convert values to string
cdf.MARRIAGE.replace({'1':'Married', '2':'Single', '3':'Other'}, inplace=True)


#-----------AGE
print(cdf.AGE.describe())
#there are ages over 120 
plt.boxplot(cdf.AGE)
plt.show()

plt.hist(cdf.AGE)
cdf.AGE.describe()

# count    23098.000000
# mean        35.703221
# std         10.273075
# min         21.000000
# 25%         28.000000
# 50%         34.000000
# 75%         41.000000
# max        141.000000
# Name: AGE, dtype: float64

#there are ages over 120 - drop these rows

cdf.AGE[cdf.AGE > 120].count()
cdf.AGE[cdf.AGE > 120]= np.nan
cdf.AGE.isnull().sum()
cdf.shape
cdf = cdf.dropna()

#create age bins
cut_bins = [0, 20, 30, 40, 50, 100]
cut_labels = ['10s', '20s', '30s', '40s', '50s']

cdf['AGEGROUP'] = pd.cut(cdf.AGE, bins=cut_bins, labels=cut_labels)
cdf.AGEGROUP.value_counts()


#-----------PAY_PC1
print(cdf.PAY_PC1.describe())
plt.hist(cdf.PAY_PC1)

#any nans?
cdf.PAY_PC1.isnull().sum()

#-----------PAY_PC2
print(cdf.PAY_PC2.describe())
plt.hist(cdf.PAY_PC2)

#any nans?
cdf.PAY_PC2.isnull().sum()

#-----------PAY_PC3
print(cdf.PAY_PC3.describe())
plt.hist(cdf.PAY_PC3)

#any nans?
cdf.PAY_PC3.isnull().sum()

#-----------AMT_PC1
print(cdf.AMT_PC1.describe())
plt.hist(cdf.AMT_PC1)

#any nans?
cdf.AMT_PC1.isnull().sum()

#-----------AMT_PC2
print(cdf.AMT_PC2.describe())
plt.hist(cdf.AMT_PC2)

#any nans?
cdf.AMT_PC2.isnull().sum()

#-----------AMT_PC3
print(cdf.AMT_PC3.describe())
plt.hist(cdf.AMT_PC3)

#any nans?
cdf.AMT_PC3.isnull().sum()

#-----------AMT_PC4
print(cdf.AMT_PC4.describe())
plt.hist(cdf.AMT_PC4)

#any nans?
cdf.AMT_PC4.isnull().sum()

#-----------AMT_PC5
print(cdf.AMT_PC5.describe())
plt.hist(cdf.AMT_PC5)

#any nans?
cdf.AMT_PC5.isnull().sum()

#-----------AMT_PC6
print(cdf.AMT_PC6.describe())
plt.hist(cdf.AMT_PC6)

#any nans?
cdf.AMT_PC6.isnull().sum()

#-----------AMT_PC7
print(cdf.AMT_PC7.describe())
plt.hist(cdf.AMT_PC7)

#any nans?
cdf.AMT_PC7.isnull().sum()

#-----------default
print(cdf.default.describe())
plt.hist(cdf.default)

#create numeric version of default

cdf['defaultnum'] = cdf.default

print(cdf.defaultnum.value_counts())
cdf.defaultnum.replace({'Y':1, 'N':0}, inplace=True)

print(cdf.defaultnum.value_counts())
cdf.defaultnum.astype('int64')


#-------------end cleaning of data

# check correlation of numeric variables
correlation = cdf.corr()

print(correlation)

sns.heatmap(correlation, square=True, cmap='RdYlGn', vmin=-1, vmax=1)
#looks like LIMIT_BAL, PAY_PC1, PAC_PC2, AMT_PC2 might have a minor relationship with default.

#need to do one hot encoding on categorical variables.

new_cdf = cdf

new_cdf = pd.get_dummies(new_cdf, columns=['SEX'], drop_first=True, prefix='SEX')
new_cdf = pd.get_dummies(new_cdf, columns=['EDUCATION'], drop_first=True, prefix='EDU')
new_cdf = pd.get_dummies(new_cdf, columns=['MARRIAGE'], drop_first=True, prefix='MAR')
new_cdf = pd.get_dummies(new_cdf, columns=['AGEGROUP'], drop_first=True, prefix='AGE')

print(new_cdf.info())

# check correlation of numeric variables
correlation = new_cdf.corr()

print(correlation)

sns.heatmap(correlation, square=False, cmap='RdYlGn', vmin=-1, vmax=1, annot=False, linewidths=0.5, xticklabels=True, yticklabels=True)
plt.xlabel('X', fontsize = 9)
plt.ylabel('Y', fontsize = 9)
plt.tick_params(labelsize=6)
plt.show()

#----------- do Logistic model regression

# get X and Y 
y = new_cdf.defaultnum.values

#X = new_cdf.drop(['defaultnum', 'ID', 'AGE', 'default'], axis=1).values
X = new_cdf.drop(['defaultnum', 'ID', 'AGE', 'default'], axis=1)

type(y)
#y = y.reshape(-1,1)
#y.shape
#y.ravel().shape

print(len(y))
print(X.count())
#------------ centre and scale data columns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        
# Create the classifier: logreg
logreg = LogisticRegression()
        
# Fit the classifier to the training data
logreg.fit(X_train, y_train)
        
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)
        
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
#AUC scores computed using 5-fold cross-validation: [0.62459126 0.62647825 0.67026845 0.66845498 0.66009284]

print("Accuracy: {}".format(logreg.score(X_test, y_test)))
#Accuracy: 0.7608695652173914

#-----------try hyperparameter tuning

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l2']}

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
        
# Fit it to the training data
logreg_cv.fit(X_train, y_train)
        
 # Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

#Tuned Logistic Regression Parameter: {'C': 1e-05, 'penalty': 'l2'}
#Tuned Logistic Regression Accuracy: 0.7574534161490684

#-----------check cross validation
cv_scores = cross_val_score(logreg, X, y, cv=5)
print(cv_scores)
#[0.75847826 0.75847826 0.75847826 0.75847826 0.75847826]


#-------------Try with scaler
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('LogReg', logreg)]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'LogReg__C':c_space,
              'LogReg__penalty':['l2']}

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

# Accuracy: 0.7965217391304348
#               precision    recall  f1-score   support

#            0       0.81      0.96      0.88      5250
#            1       0.68      0.28      0.40      1650

#     accuracy                           0.80      6900
#    macro avg       0.75      0.62      0.64      6900
# weighted avg       0.78      0.80      0.76      6900

# Tuned Model Parameters: {'LogReg__C': 31.622776601683793, 'LogReg__penalty': 'l2'}

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# [[5034  216]
#  [1188  462]]
#               precision    recall  f1-score   support

#            0       0.81      0.96      0.88      5250
#            1       0.68      0.28      0.40      1650

#     accuracy                           0.80      6900
#    macro avg       0.75      0.62      0.64      6900
# weighted avg       0.78      0.80      0.76      6900


# Compute predicted probabilities: y_pred_prob
y_pred_prob_cv = cv.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr_cv, tpr_cv, thresholds_cv = roc_curve(y_test, y_pred_prob_cv)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_cv, tpr_cv)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Compute cross-validated AUC scores: cv_auc
cv_auc2 = cross_val_score(cv, X, y, cv=5, scoring='roc_auc')
# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc2))
#AUC scores computed using 5-fold cross-validation: [0.73592897 0.73476999 0.74804845 0.74862284 0.76429019]

print("Accuracy: {}".format(cv.score(X_test, y_test)))

#outcome 
#- Better AUC scores when scaling - 0.73592897 compared with 0.62
#- Better accuracy when scaling 0.80 compared with 0.79

#------------------Using NN
import tensorflow as tf

# X_train_tf = tf.convert_to_tensor(X_train)
# y_train_tf = tf.convert_to_tensor(y_train)
# X_test_tf = tf.convert_to_tensor(X_test)
# y_test_tf = tf.convert_to_tensor(y_test)

# type(X_train_tf)
# type(y_train_tf)

# print(X_train_tf.shape)

#X_train.info()
#y_train[:5]

# y_train_df = pd.DataFrame({'Target': y_train[:]})
# y_train_df.info()


train_tf = tf.data.Dataset.from_tensor_slices((X_train, y_train))

type(train_tf)


#for feat, targ in train_tf.take(1):
# print ('Features: {}, Target: {}'.format(feat, targ))

# train_dataset = train_tf.shuffle(len(X_train)).batch(1)
# print(X_train_tf.shape)



def apply_scaler(df):
    SS = StandardScaler() 
    df[['LIMIT_BAL', 'PAY_PC1', 'PAY_PC2','PAY_PC3','AMT_PC1','AMT_PC2','AMT_PC3','AMT_PC4','AMT_PC5','AMT_PC6','AMT_PC7']] = SS.fit_transform(df[['LIMIT_BAL', 'PAY_PC1', 'PAY_PC2','PAY_PC3','AMT_PC1','AMT_PC2','AMT_PC3','AMT_PC4','AMT_PC5','AMT_PC6','AMT_PC7']])

    return df
    
def split_train_test(df):
    # get X and Y 
    y = df.defaultnum.values

    #X = new_cdf.drop(['defaultnum', 'ID', 'AGE', 'default'], axis=1).values
    X = df.drop(['defaultnum', 'ID', 'AGE', 'default'], axis=1)

    # Create training and test sets
    fX_train, fX_test, fy_train, fy_test = train_test_split(X, y, test_size = 0.3, random_state=42)

    return fX_train, fX_test, fy_train, fy_test
    

scaled_cdf = apply_scaler(new_cdf)

scaled_cdf.info()

X_train, X_test, y_train, y_test = split_train_test(scaled_cdf)


#---------------type 1

model = ""
num_epochs = 100

model = tf.keras.Sequential()

#input shape is the number of features (columns)
model.add(tf.keras.layers.Dense(21, activation='sigmoid', input_shape=(21,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#Set the optimizer, loss function, and metrics
#model.compile(optimizer='Adam', loss='huber_loss', metrics=['accuracy'])
model.compile(optimizer='SGD', loss='huber_loss', metrics=['accuracy'])


# Add the number of epochs and the validation split
tf_hist = model.fit(X_train, y_train, epochs=num_epochs, validation_split=0.20)
#model.fit(train_dataset, epochs=10, validation_split=0.20)

model.evaluate(X_test, y_test)
#accuracy- 0.75


#plot model accuracy
tf_test_loss = tf_hist.history['val_loss']
tf_train_loss = tf_hist.history['loss']
tf_train_acc = tf_hist.history['acc']
tf_test_acc = tf_hist.history['val_acc']


plt.subplot(1,2,1)
plt.plot(range(num_epochs), tf_train_loss, color='blue', label='train')
plt.plot(range(num_epochs), tf_test_loss, color='green', label='test')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.plot(range(num_epochs), tf_train_acc, color='blue', label='train')
plt.plot(range(num_epochs), tf_test_acc, color='green', label='test')
plt.ylabel('accuracy')
plt.legend(loc='upper right')


plt.show()
plt.clf()


# Compute and print the confusion matrix and classification report

tf_train_prob = model.predict(X_train)
tf_test_prob = model.predict(X_test)

#mydf = pd.DataFrame({'Prob': tf_train_prob[:,0]})
#mydf.describe()

tf_train_prob[tf_train_prob <= 0.5]=0
tf_train_prob[tf_train_prob > 0.5]=1

tf_test_prob[tf_test_prob <= 0.5]=0
tf_test_prob[tf_test_prob > 0.5]=1


print(confusion_matrix(y_test, tf_test_prob))
print(classification_report(y_test, tf_test_prob))


#res_df = pd.DataFrame({'train_pred': tf_train_prob, 'train_act': y_train})
#plt.hist(tf_train_prob)
#plt.show()




#---------------type 2

# def get_compiled_model():
#   model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(1)
#   ])

#   model.compile(optimizer='adam',
#                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
#   return model

# model = get_compiled_model()
# model.fit(train_dataset, epochs=15)


print(model.summary())






