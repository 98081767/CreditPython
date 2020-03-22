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

cdf = pd.read_csv('Credit card\\AT3_credit_train_STUDENT.csv')

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

sns.heatmap(correlation, square=True, cmap='RdYlGn', vmin=-1, vmax=1)
#looks like LIMIT_BAL, PAY_PC1, PAC_PC2, AMT_PC2 might have a minor relationship with default.

#need to do one hot encoding on categorical variables.




