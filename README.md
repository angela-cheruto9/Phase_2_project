# Phase_2_project

## Data preparation

* import necessary libraries*
>import pandas as pd
>>import numpy as np
>>>import matplotlib.pyplot as plt
>>>>import seaborn as sns
>>>>>%matplotlib inline

*load the dataset*
df = pd.read_csv("kc_house_data.csv")
df.head(10)

## Exploratory data analysis
### Data cleaning
*check the number of rows and columnns *
df.shape

*checkin the datatypes in the dataset*
df.info()

*checking for unique data in the dataset*
df.nunique()

*checking why sqft_basement is an object*
df.sqft_basement.unique()

*dropping the rows that have '?'*
df = df[df.sqft_basement != '?']

*converting the sqft_basement column from str to float*
df['sqft_basement'] = df['sqft_basement'].astype('float')

*displaying values in float* 
pd.set_option('display.float_format', lambda X: '%.5f'% X)

*check the statistic summary of the dataset*
df.describe()

*descriptive statistics of the dataset*
df.describe().transpose()

### checking for null values
*checking for number of null values in each column in the dataset*
df.isnull().sum().sort_values(ascending =False)

*checking for number of null values in each row *
df.isnull().sum(axis=1).sort_values(ascending =False)

*checking for percentage of null values in each row *
df.isnull().sum(axis=1).sort_values(ascending =False)/len(df)*100

### working on columns with null values
*checking statistic summary of the data in the view column*
df.view.describe()

*replacing the null values in the view column with the median*
df.view = df.view.fillna( value = df.view.median())

*checking statistic summary of the data in the waterfront column *
df.waterfront.describe()

*checking the value_count of properties with a waterfront*
df.waterfront.value_counts()

* replacing the null values in the waterfront column with the median *
df.waterfront = df.waterfront.fillna( value = df.waterfront.median())

*checking statistic summary of the data in the yr_renovated column *
df.yr_renovated.describe()

* replacing the null values in the waterfront column with the median *
df.yr_renovated = df.yr_renovated.fillna( value = df.yr_renovated.median())

* recheck if there are any missing values *
df.isnull().sum()

*drop unnecessary columns*
df.drop('id', axis=1, inplace=True)

*inspecting the null values using a heatmap*
sns.heatmap(df.isnull(),yticklabels = False ,cbar= False,cmap = 'coolwarm')

## Correlation of features
*correlation of features*
corr = df.corr()
corr

*change correlation to True for positive or negative correlations that are bigger than 0.75 in the correlation matrix*
abs(df.corr()) > 0.75

*heatmap to show correlation of features*
plt.figure(figsize =(12,12))
sns.heatmap(corr,cbar=True,square= True ,fmt = '.1f', annot=True,annot_kws={'size':10},cmap= 'viridis')

## Visualization
*Histogram to represent the variables in the dataset *

df.hist(figsize=(20,15), bins='auto');

## Sorting data 
*sorting values of price in descending order to check the first 10 rows*
df.sort_values(('price'), ascending =False).head(10)

*sorting values of price in descending order to check the last 10 rows*
df.sort_values(('price'), ascending =False).tail(10)

*checking the value_count of price of the houses in the dataset*
df['price'].value_counts()

*checking the value_count for each variable in the dataset *
for value in df:
    print(df[value].value_counts())
    print()
    
* displot to show the price distribution *
plt.figure(figsize = (20,18))
sns.displot(df['price'])

*convert the date column so that pandas recognizes it as a date *
df['date'] = pd.to_datetime(df['date'])

*create a colum to hold the month data extracted from the date column,in order to find out if the seasons(summer,winter,spring,autumn)affect house prices*
df['month'] = df['date'].dt.month

## Checking the relationship of data in the dataset

*scatterplot to show the distribution of waterfront against price*
plt.figure(figsize = (10,6))
sns.scatterplot(x = 'month' , y ='price', data=df , hue ='waterfront')

*scatterplot to show the distribution of waterfront against price*
plt.figure(figsize = (10,6))
sns.scatterplot(x='long' , y ='lat', data=df , hue ='price')

*scatterplot to show the distribution of waterfront against price*
plt.figure(figsize = (10,6))
sns.scatterplot(x = 'month' , y ='price', data=df , hue ='view')

*displot to show the distribution of bedrooms*
plt.figure(figsize = (10,6))
sns.countplot(df['bedrooms'])

*boxplot to show the distribution of bedrooms against price*
plt.figure(figsize = (10,6))
sns.boxplot(x='bedrooms' , y ='price', data=df)


*displot to show the distribution of bathrooms*
plt.figure(figsize = (15,6))
sns.countplot(df['bathrooms'])


*plot a boxplot to show the distribution of bathrooms against price*
plt.figure(figsize = (15,6))
sns.boxplot(x='bathrooms' , y ='price', data=df)

*histogram to show grade distribution*

data= df['grade']
plt.figure(figsize=(10,5))
plt.hist(data, bins=30, align='left', color='b', edgecolor='black',
              linewidth=1)
 *axis labels*
plt.xlabel("grade")
plt.xticks(rotation= 80,fontsize=10)
plt.ylabel("Frequency of grade")
plt.title("Distribution of grade in the dataset")
 
plt.show()

*boxplot to show the distribution of grade against price*
plt.figure(figsize = (10,6))
sns.boxplot(x='grade' , y ='price', data=df)

*histogram to show the yr_built column distribution*
*plot a histogram to show the distribution of yr_built*
data= df['yr_built']
plt.figure(figsize=(10,5))
plt.hist(data, bins=30, align='left', color='b', edgecolor='black',
              linewidth=1)
 *Add axis labels*
plt.xlabel("yr_built")
plt.xticks(rotation= 80,fontsize=10)
plt.ylabel("Frequency of yr_built")
plt.title("Distribution of yr_built in the dataset")
 
plt.show()

*displot to show the distribution of floors*
plt.figure(figsize = (10,6))
sns.countplot(df['floors'])

*Scatter plot to show the correlation of the sqft_living against price*
plt.figure(figsize = (10,6))
sns.scatterplot(x='sqft_living', y='price', data=df)

*Scatter plot to show the correlation of the sqft_lot against price*
plt.figure(figsize = (10,6))
sns.scatterplot(x='sqft_lot', y='price', data=df)

*Scatter plot to show the correlation of the sqft_lot against price*
plt.figure(figsize = (10,6))
sns.scatterplot(x='condition', y='price', data=df)

*boxplot to show the distribution of waterfront against price*
plt.figure(figsize = (10,6))
sns.boxplot(x='waterfront' , y ='price', data=df)

## Splitting data into categorical and numerical data
*creating new variables to hold categorical data*
df_categorical =df[['view','condition','waterfront','month','yr_built','yr_renovated','zipcode']]

*creating new variables to hold the numerical data*
df_numerical=df[['price','bedrooms','bathrooms','floors','sqft_lot','sqft_basement','sqft_above','lat','long','sqft_living15','sqft_lot15']]                 
*create bins*
bins = [0, 1, 4]
*split view column into viewed and unviewed houses*
labels = ['unviewed','viewed']
bins_viewed = pd.cut(df['view'], bins=bins, include_lowest = True,labels=labels)
bins_viewed = bins_viewed.cat.as_unordered()

*value_count for viewed and unviewed houses*
views = bins_viewed.value_counts()
views

*bargraph to show the total viewed and unviewed houses*
views.plot(kind='bar')

*summary statistics of categorical variables*
df_categorical('yr_renovated').describe()

* create bins to hold the column data*
bins = [0, 1, 2015]
*split yr_renovated column to renovated and not_renovated houses*
labels = ['not_renovated','renovated']
renovated_bins = pd.cut(df['yr_renovated'], bins=bins,include_lowest = True, labels=labels)
renovated_bins = renovated_bins.cat.as_unordered()

*value_count for renovated and not_renovated bins*
renovated_bins.value_counts().plot(kind='bar')

*drop yr_renovated columns *
df_categorical = df_categorical.drop(columns=['view','yr_renovated'])
*replace the dropped columns with the binned data*
df_categorical= pd.concat([df_categorical, bins_viewed,renovated_bins], axis=1)

*checking the categorical data*
df_categorical.info()

## One hot encoding
*hot encoding data for prediction*
df_categorical = pd.get_dummies(df_categorical) 
df_categorical.info()

*convert condition and waterfront to int*
df_categorical['condition'] = df_categorical['condition'].astype("int") 
df_categorical['waterfront'] = df_categorical['waterfront'].astype("int")

*converting the categorical columns to string to enable prediction*
df_categorical['condition'] = df_categorical['condition'].astype("str") 
df_categorical['waterfront'] = df_categorical['waterfront'].astype("str")
df_categorical['zipcode'] = df_categorical['zipcode'].astype("str")

*hot encoding the data again*
df_categorical = pd.get_dummies(df_categorical) 
df_categorical.info()

## Standardization/normalization
### mean normalization

* import standardscaler*
from sklearn.preprocessing import StandardScaler
*referencing the standardscaler*
std = StandardScaler()
std

*creating a variable for scaled data*
*fitting and transforming data*
df_std = pd.DataFrame(std.fit_transform(df_numerical),columns = df_numerical.columns) 
df_std.head()

*summary statistics of numerical data*
df_std.describe()

*Histogram to represent all numerical data in the dataset *
df_std.hist(figsize = (20,20), color = "b", lw=0); 

## min-max normalization
*import min-max scaler*
from sklearn.preprocessing import MinMaxScaler 
*referencing the scaler*
mmscaler = MinMaxScaler() 
df_minmax = pd.DataFrame(mmscaler.fit_transform(df_numerical),columns = df_numerical.columns) 

*Histogram to represent all numerical data in the dataset* 
df_minmax.hist(figsize = (20,20), color = "b");

## log transformation

import warnings
warnings.filterwarnings('ignore')

*create a dataframe to hold the data from df_numerical*
df_price_log= pd.DataFrame([]) 
df_price_log = df_numerical

*price log transformation*
df_price_log['price']= np.log(df_price_log['price'])

*manually performing mean normalization on the rest of our varaibles*
df_price_log['bathrooms']= (df_price_log['bathrooms'] - df_price_log['bathrooms'].mean())/df_price_log['bathrooms'].std()
df_price_log['bedrooms']= (df_price_log['bedrooms'] - df_price_log['bedrooms'].mean())/df_price_log['bedrooms'].std()
df_price_log['floors']= (df_price_log['floors'] - df_price_log['floors'].mean())/df_price_log['floors'].std()
df_price_log['sqft_lot']= (df_price_log['sqft_lot'] - df_price_log['sqft_lot'].mean())/df_price_log['sqft_lot'].std()
df_price_log['sqft_above']= (df_price_log['sqft_above'] - df_price_log['sqft_above'].mean())/df_price_log['sqft_above'].std()
df_price_log['lat']= (df_price_log['lat'] - df_price_log['lat'].mean())/df_price_log['lat'].std()
df_price_log['long']= (df_price_log['long'] - df_price_log['long'].mean())/df_price_log['long'].std()
df_price_log['sqft_living15']= (df_price_log['sqft_living15'] - df_price_log['sqft_living15'].mean())/df_price_log['sqft_living15'].std()
df_price_log['sqft_lot15']= (df_price_log['sqft_lot15'] - df_price_log['sqft_lot15'].mean())/df_price_log['sqft_lot15'].std()

df_price_log.head()

*histogram showing price distribution*
df_price_log['price'].hist() 

## KDE plots
*from our normalized data create a list of columns*
data = list(df_price_log) 
*checking the list of columns using for loop*
for column in data:
    # create a histogram
    df_price_log[column].plot.hist(density=True, label = column+' histogram' ) 
    # create a KDE plot
    df_price_log[column].plot.kde(label =column+' kde') 
    plt.legend()
    plt.show() 
    
# jointplots
*from the list of normalized data create a joint plot*
import warnings
warnings.filterwarnings('ignore')
*access each column in the data*
for column in data:
    sns.jointplot(x=column, y="price", #creating our joint plot as well as setting our columns to be predictors and price to be our target
                  data= df_price_log, #we want the jointplots to be created using our df_price_log dataset
                  kind='reg', 
                  label=column, 
                  joint_kws={'line_kws':{'color':'red'}}) #stylistic choices

plt.legend() 
plt.show()

*joining categorical data to numerical data*
df_num_cat = pd.concat([df_price_log, df_categorical], axis=1) 
df_num_cat.head()

*checking data types in the column*
df_num_cat.dtypes

*converting categorical data to category*
for categorical_names in df_num_cat.iloc[:,11:].columns:
         df_num_cat[categorical_names] = df_num_cat[categorical_names].astype('category')

## Modelling data
### OLS regression
*importing the required libraries*
import statsmodels.api as sm #importing the necessary libraries
import statsmodels.formula.api as smf

*estimate model*
df = df_num_cat
f = 'price ~  bedrooms + bathrooms + floors + sqft_lot + sqft_basement +sqft_above + lat + long + sqft_living15 ' 
*fitting the model*
model = smf.ols(formula=f, data=df_num_cat).fit()
*model summary*
print(model.summary())

## Linear regression using sklearn
*sample target and independent variable *
*using sklearn.model_selection.train_test_split*
X = df_num_cat.drop(['price'],axis = 1)
y = df_num_cat['price']

*import library*
from sklearn.linear_model import LinearRegression
*Creating a linear regressor*
linearegression = LinearRegression()
*Training the model*
linearegression.fit(X, y)

*y intercept value*
linearegression.intercept_

*coefficients*
linearegression.coef_

*checking the pvalues associated with coefficients *
model.pvalues

# Model training
## Perfoming train-test split

*Splitting of data in train and test*
from sklearn.model_selection import train_test_split
*using 30% of the total data for testing*
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3 , random_state = 40)

* inspecting X_train*
X_train

*checking the datatypes in X_train*
X_train.info()

*Checks if there are any null values8
X_train.isna().sum()

*Inspecting y_train*
y_train

*inspecting data in X_test *
X_test

*checking the datatypes in X_test*
X_test.info()

* check if there are any null values*
X_test.isna().sum()

*checking what y_test contains*
y_test

# Prediction
*importing required library*
from sklearn.linear_model import LinearRegression
linearegression = LinearRegression ()

* fitting the model*
linearegression.fit(X_train,y_train)

*assigning X_test prediction to a variable*
y_pred1 = linearegression.predict(X_test)
y_pred1

*inspecting y_test*
y_test

*inspecting X_test*
X_test

# Calculating term errors

*importing the necessary library*
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred1)
r_squared = r2_score(y_test, y_pred1)

*output the mean_squared_error and r_squared_value*
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)

*checking for P-value using statsmodel*
import statsmodels.api as sm
X_train_sm = X_train
*Unlike SKLearn, statsmodels don't automatically fit a constant,* 
*so you need to use the method sm.add_constant(X) in order to add a constant* 
X_train_sm = sm.add_constant(X_train_sm)
 *create a fitted model in a single line*
lm_1 = sm.OLS(y_train,X_train_sm).fit()

* print the coefficients*
lm_1.params

*summary of the model*
print(lm_1.summary())

# Final model
*creating a new dataframe "df_final" that has columns that will be retained*
*dropping columns*
final_model = df_num_cat.drop(["sqft_lot","long","sqft_lot15","yr_built","month"], axis=1)

*final model*
final_cols = final_model.drop(['price'],axis=1) #let's see what our final model looks like
formula = 'price ~ ' + ' + '.join(final_cols) 
model = smf.ols(formula=formula, data = final_model) 
res = model.fit()
*getting the model summary*
res.summary()

                       
                       













