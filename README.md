# Predicting-Bankruptcy-of-Firms
This repository contains all the files of the Bankruptcy Prediction Project, related to my Data Mining class.

![Bank Icon](images/bank_icon.png)
# Table of Contents
- [Overview](#overview)
- [Data Dictionary](#data-dictionary)
- [Variable Descriptions](#variable-descriptions)
- [Methodology](#methodology)
  - [1. Data Processing](#1-data-pre-processing)
    - [Feature Elimination](#feature-elimination)
    - [Missing Value Imputation](#missing-value-imputations-mvi)
  - [2. Feature Engineering](#2-feature-engineering)
    - [Name Analysis](#name-analysis)
    - [Group Variables](#group-categories)
    - [Weight of Evidence](#weight-of-evidence)
  - [3. Modeling](#3-modeling)
    - [Train & Validation Split](#train-and-validation-split)
    - [Train Multiple Models](#train-multiple-models)
    - [Model Selection](#model-selection)
  - [4. Scoring](#4scoring)
  
## Overview
Firm collapse prediction has been a subject of interest, for almost a century, and it still ranks high among the hottest topics in economics. The aim of predicting financial distress is to develop a predictive model that combines various econometric measures and allows to foresee a financial condition of a firm. The purpose of the bankruptcy prediction is to assess the financial condition of a company and its future perspectives within the context of long-term operation on the market.

## Data Dictionary
We have been provided with a ```Train (10,000 x 65)``` & ```Test (5000 x 64)``` datasets. The dependant variable is `"Class"`, It is a binary variable indicating 1 for Bankrupted firms and 0 otherwise.
![Data Dictionary](images/)

## Variable Descriptions
We can see the ```variable descriptions``` below
![Variables](images/)

## Methodology
Now that we have understood what the problem statement is, let us follow a methodology to solve this. 

### 1. Data sampling

![Missing](images/Missing_Train.PNG)

#### Feature Elimination
- Drop ```cabin``` variable due to high missing percentage
- Drop ```fare``` because of **high correlation** with ```PClass```
- Drop ```Ticket``` variable due to low value addition
#### Missing Value Imputations (MVI)
We have already dropped the ```cabin``` variable, so we have to impute the ```Age``` and ```Embarked``` variables. Age is a continuous variable and its distribution is below. From the figure we can say that ```Age``` is skewed.
![Age](images/Age_Dist.PNG)

I have used the median value of ```Age``` for a `passenger class` and `gender` to impute the missing values.

### 2. Data Exploration
This is an import aspect of the methodology, because this is where the business intuition and domain expertise come in. And we all know how crucial these two are to make better predictions and to interpret the results of the model.  

#### Name Analysis
Though this variable `might not look important` at first, but we can extract some `hidden information` from this i.e., we can get the `Title` of each passenger and analyze if some `titles` have high survival probability. We have `Capt`, `Col`, `Don`, `Dr`, `Jonkheer`, `Lady`, `Major`, `Rev`, `Sir`, `the Countess`, `Miss`, and `Mrs`
#### Group Categories
I have now `grouped some categories` together because they have the `same event rates` i.e., `same probability of survival`.
#### Weight of Evidence
These categories have string values, so we need to convert them to numerical values but with the same degrees of freedom. So we use `Weight of Evidence` approach to accomplish this. 
##### What is Weight of Evidence (WOE)?
The weight of evidence tells the `predictive power` of an independent variable in relation to the dependent variable. It is generally described as a measure of the `separation of goods (Survived = 1)` and `bads (Survived = 0)`. We use the below formula to calculate the WOE and I have also provided a sample example for explanation purposes.

![WOE](images/woe.png)

#### WOE Calculation for a variable with values `Category 1` & `Category 2`
```python
{
"""
#---------------------#
#   WOE & IV Example  #
#---------------------#

#-------------------#
#   Total           #
#-------------------#                       
Total Rows = 1000 || Total Goods = 300 || Total Bads = 700

#-------------------#                  |       #-------------------#
#   Category 1      #                  |       #   Category 2      #
#-------------------#                  |       #-------------------# 
All = 610 || Goods = 160 || Bads = 450 | All = 390 || Goods = 140 || Bads = 250

#--------------#
#   WOE & IV 1 #
#--------------#    
%Goods = (160/300) & %Bads = (450/700) => WOE 1 = ln(%Goods / %Bads) => IV 1 = WOE 1 * (%Goods - %Bads)

#--------------#
#   WOE & IV 2 #
#--------------#    
%Goods = (140/300) & %Bads = (250/700) => WOE 2 = ln(%Goods / %Bads) => IV 2 = WOE 2 * (%Goods - %Bads)
"""
}
```
### 3. Data Preparation
We have `cleaned` the data and `derived` some variables so that we can make better predictions. So let us `predict` now. But we need to follow some steps to make a robust model and `avoid over-fitting` the data.

#### Train and Validation Split
The training data will be `randomly` split into `70:30` ratio into `training` and `validation` datasets. We now use the first one to train our model, and the validation data to validate our model's accuracy.
#### Train Multiple Models
I have explored `six` different techniques to train the model. Click on the links for literature review.
- [Logistic Regression](https://www.analyticsvidhya.com/blog/2021/03/logistic-regression/)
- [Support Vector Machines](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)
- [Decision Trees](https://www.analyticsvidhya.com/blog/2016/04/tree-based-algorithms-complete-tutorial-scratch-in-python/)
- [Random Forest](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-random-forest-and-its-hyper-parameters/)
- [Light Gradient Boosting](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/)
- [Extreme Gradient Boosting](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/)
#### Model Selection
The performance of the above models can be judged based on the validation dataset. The results are below, so my best model is Light GBM.
```python
{
"""
Logit model validation Accuracy: 82.46%
SVM model validation Accuracy: 82.84%
DT model validation Accuracy: 83.58%
RF model validation Accuracy: 83.58%
LGB model validation Accuracy: 85.82%
XGB model validation Accuracy: 82.84%  
""" 
}
```
### 4.Model Development
We now have a model, trained and validated. Recollect that we have been provided a `test` dataset to make predictions for the `future`. So we perform the same `data-preprocessing` steps on this as well and predict the `Survived` column. But, for this we can `train` our model on the `whole training` dataset and again and use that model so that we have more data to train our model.

We now `submit` the predictions and the `leaderboard score` tells the accuracy we have obtained on the test data. This whole modeling process is an `iterative` one because a `huge number parameters` are involved in the whole lifecycle.

This project has been a great starting point for me. Hopefully it is the same for the readers as well. Thanks!

### 4.Model Comparison