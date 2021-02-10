# Toxic Comment Classification Challenge
Identify and classify toxic online comments

| Name | Type  | Public Kernel | Metric | Date |
| ------ | ------ | ------ | ------ | ------ | 
|[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)| NLP, Classification | [1. Logistic Regression TFIDF ](https://www.kaggle.com/sudhirnl7/logistic-regression-tfidf), [2. Logistic regression with hashing vectorizer](https://www.kaggle.com/sudhirnl7/logistic-regression-with-hashing-vectorizer) | AUC | Dec-2017|

In this competition, you’re challenged to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate

## Data set Description
We are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

toxic
severe_toxic
obscene
threat
insult
identity_hate
We must create a model which predicts a probability of each type of toxicity for each comment. train data of size 160000 observations and test data of size 153000 observations.

## Feature Engineering
TFIDF term frequency inverser document frequency is used to extract feature from text data.

## Modeling
Logistic regression is used to build model

## Model validation
Submissions are now evaluated on the mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column.

## Out come of project:
This project give good opportunity to explore data set. Create new feature from data  set.
