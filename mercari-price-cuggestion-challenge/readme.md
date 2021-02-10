# Mercari Price Suggestion Challenge
Can you automatically suggest product prices to online sellers?
| Name | Type  | Public Kernel | Metric | Date |
| ------ | ------ | ------ | ------ | ------ | 
| [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge) | Regression | NA | RMSLE | Nov 2017 |

It can be hard to know how much something’s really worth. Small details can mean big differences in pricing.Product pricing gets even harder at scale, considering just how many products are sold online. Clothing has strong seasonal pricing trends and is heavily influenced by brand names, while electronics have fluctuating prices based on product specs.
In this competition, we need to build an algorithm that automatically suggests the right product prices. We’ll be provided user-inputted text descriptions of their products, including details like product category name, brand name, and item condition.

## Data set
train.tsv, test.tsv
name - the title of the listing.category_name, brand_name, shipping
item_condition_id - the condition of the items provided by the seller
price - the price that the item was sold for. This is the target variable that you will predict. The unit is USD.
item_description - the full description of the item

## Evaluvation metrics
The root mean square log error (RMSLE) is used as evaluvation metric to analyze performance of model. 

RMSLE = $\sqrt {\frac{1}{n}\sum_{1}^{m}(log(y'+1) +log(y+1))}$

m  is the total number of observations 
y' is your prediction
y is the actual value 
log(x) is the natural logarithm of x


## Feature engineering
CountVectorizer,TfidfVectorizer,LabelBinarizer,One hot encoding.

## Model
The ridge (l2 regularized linear regression) methods is used to build model.

## Out come of project
This project help to analysze property of atoms using exploratory data analysis.
