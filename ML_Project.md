---
title: "Practical Machine Learning Project for Weight Lifting Exercise Predictions"
author: "Javier Segovia"
date: "24/07/2014"
output: html_document
---

## Including Libraries


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(kernlab)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

## Raw Data

Here the training and final test is read as raw data from 2 different csv files.


```r
raw.tr<-read.csv("pml-training.csv")
raw.fts<-read.csv("pml-testing.csv")
```

## Preprocessing

### Data Partition

In order to have a reproducible experiment, the seed "6789" is assigned, this will produce the same pseudo-random numbers list to create the data partition of the raw training set, to split it into a new training set (80% of the cases) and a new test set (20% of the cases) used for cross-validation.


```r
set.seed(6789)
classe <- raw.tr$classe
it<-createDataPartition(classe, p = 0.8, list = FALSE)
tr<-raw.tr[it,]
ts<-raw.tr[-it,]
```

### Zero or Near-Zero Variance

When a variable has zero or near zero variance means that is not relevant to be used in a predictor model, because different outcomes will have similar values on that variables, so the can be deleted from the training and test set, creating in that way an easier set to train the model.


```r
nzv <- nearZeroVar(tr)
tr<-tr[-nzv]
ts<-ts[-nzv]
fts<-raw.fts[-nzv]
```

### Numeric Columns

Then, to build a regression model, numerical columns are required but in the training set there is also other classes of variables like integers or factors that can lead to misclassification of the variable "classe" in the predictor, so the approach is to avoid these columns. 


```r
ok.cols = which(lapply(tr, class) %in% c("numeric"))
tr   <-tr [,ok.cols]
ts   <-ts [,ok.cols]
fts  <-fts[,ok.cols]
```

### Imputing Missing Values

There are various preProcess methods that impute missing values in the data sets to have a better regression model, but it might introduce some bias. K-nearest neighbor ("knnImpute") finds the k closest samples, bagging fits a bagged tree model for each predictor ("bagImpute") it is very slow but has high accurancy, and the medians ("medianImpute") is fast but treats each predictor independently and may be inaccurate. Thus, the choice is to use "knnImpute" for pre-processing the data.


```r
pp         <- preProcess(tr, method = c("knnImpute"))
transf.tr  <- data.frame(classe = classe[it] , predict(pp, tr))
transf.ts  <- data.frame(classe = classe[-it], predict(pp, ts))
transf.fts <- predict(pp, fts)
```


## The Model

There are a lot of models to be chosen in ML, but the one used for this project is the random forest explained in the subject videos and slides.

### Random Forest

It implements the Breiman's random forest algorithm for classification and regression, and the way it is used is the function "randomForest", using the formula "classe ~ ." that means the classe is the outcome of the prediction and the rest of variables are the predictors, and the data is the transformed training set that is the training data after the preprocessing. The total number of trees in the forest is 100 and the number of random variables for each tree is 10 (to avoid overfitting).


```r
mf.RF<-randomForest(classe ~ ., transf.tr, ntree = 100, mtry = 10)
```

## Cross-Validation
The idea in CV is that the model fits correctly the training set and avoid overfitting for the test set, to use a model that predicts with high accuracy. With tha random forest model it must generalize well the problem due to the quantity of trees used in the forest and an acceptable number of random variables.

### In-sample Error
The predictions in the training set using the random forest model has an accuracy of 100%. It is a good result for the training data but it is possible to have overfitting, so the out-of sample error must be checked to know if the model generalizes correctly or not.


```r
prediction.tr<-predict(mf.RF, transf.tr)
print(confusionMatrix(prediction.tr, transf.tr$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

### Out-of-sample Error

Thus, the out-of-sample error for the test set confirms with a 99.1% of accuracy that the model generalizes correctly, so it can be used for the final test set to predict the final results.


```r
prediction.ts<-predict(mf.RF, transf.ts)
print(confusionMatrix(prediction.ts, transf.ts$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1110    3    1    1    0
##          B    4  750    6    2    1
##          C    0    6  676    7    1
##          D    0    0    1  631    0
##          E    2    0    0    2  719
## 
## Overall Statistics
##                                         
##                Accuracy : 0.991         
##                  95% CI : (0.987, 0.993)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.988         
##  Mcnemar's Test P-Value : 0.146         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.995    0.988    0.988    0.981    0.997
## Specificity             0.998    0.996    0.996    1.000    0.999
## Pos Pred Value          0.996    0.983    0.980    0.998    0.994
## Neg Pred Value          0.998    0.997    0.998    0.996    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.283    0.191    0.172    0.161    0.183
## Detection Prevalence    0.284    0.194    0.176    0.161    0.184
## Balanced Accuracy       0.996    0.992    0.992    0.991    0.998
```

## Results

Finally, the testing file predicted with the randomForest model lead to a 100% accuracy over the small test set whose solutions are printed below, that solutions have been separated into different files the provided function "pml_write_files" due to the independent submissions of each case for the programming part of the project.


```r
answers<-predict(mf.RF, transf.fts)
print(answers)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```
