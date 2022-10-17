# King's county housing

![image](https://user-images.githubusercontent.com/104419035/196160831-41caa804-db38-4543-9594-c00f68a27936.png)


## Overview

Build a regression model that will be used to analyze house sales in a northwestern county.

## Data

This project uses the King County House Sales dataset, which can be found in [kc_house_data.csv](https://github.com/angela-cheruto9/Phase_2_project/blob/master/kc_house_data.csv) in the data folder in this repo. The description of the column names can be found in [column_names.md](https://github.com/angela-cheruto9/Phase_2_project/blob/master/column_names.md) in the same folder. 

## Defining Experimental design

* Importing the relevant libraries used in the analysis.

* Loading data

* Explore the dataset we will use for our project.

* Exploratory Data Analysis (EDA)

* Data Pre-processing

* Modelling and Evaluation

* Challenging the model

* Conclusion

* Recommendations


### Libraries

Matplotlib

Seaborn

Scikit-Learn

## Data exploration

The dataset contains 21597 rows and 21 columns. 
The dataset also contains both continuous and categorical data, several time based columns like yr_built and yr_renovated and  categorical data like the grade, condition.

Below is a visualization summary of the variables present in the dataset.

![Screenshot (390)](https://user-images.githubusercontent.com/104419035/196168709-d3c9000a-3b27-479b-9628-281feb8d9cad.png)

![Screenshot (391)](https://user-images.githubusercontent.com/104419035/196168922-27540604-b7a9-4f13-8abf-d0c8454bc05d.png)

## Data preparation

Checking and working on null values
Checking correlation of features and removing multicollinearity
Datatype conversion
checking relationship of data in the dataset
Splitting data into categorical and numerical variables
One-hot encoding
Standardization of numerical data

## Modelling and evaluation
Assigned variables to X and y (target variable) which is price and performed a train-test-split.
I then built a linear regression model to predict the price of the houses

## Conclusion

*  Most predictors have a positive relationship with price
*  Sqft-above had the highest slope followed by sqft_living15.
*  The sqft_above(square footage of house apart from basement) seems to be the one that matters most when it came to price. 
*  A slight increase in sqft_above space there is a significant increase in price.
*  Sqft_living15 seemed to also be a major aspect of prediction as it has a strong positive relationship with price as we could see in the slope of the jointplot
*  There was no negative relationship betweeen predictors and target variable
*  The final model hads predictors with a p_value of less than 0.05 and an r_squared value of 0.7
* final prediction variables include sqft_above,sqft_living15,bathrooms,sqft_basement,bedrooms,floors,view

## More Information
For detailed information kindly refer to our Notebook and presentation slides found in the [Github repository](https://github.com/angela-cheruto9/Phase_2_project).











                       
                       













