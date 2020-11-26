#Sberbank
library(randomForest)
library(gbm)
library(dplyr)
library(caret)

#Import data
setwd('/home/sudhir/R/Sberbank')
train<-read.csv('train.csv')
test<-read.csv('test.csv')
macro<-read.csv('macro.csv',na.strings = 'NA')

test$price_doc=0
#Feature engineering
full0<-bind_rows(train,test)
full0$timestamp<-as.Date(full0$timestamp)
macro$timestamp<-as.Date(macro$timestamp)
full0<-merge(full0,macro,by='timestamp')

#Feature engineering
table(is.na(full0))
colSums(is.na(full0))

#target variable
summary(full0$price_doc)
#near zero variance for numeric variable
num_var0<-names(full0)[which(sapply(full0,is.numeric))]
cat_var0<-names(full0)[which(sapply(full0,is.factor))]

nzv<-nearZeroVar(full0[num_var0],saveMetrics = TRUE)
dim(nzv[nzv$percentUnique>0.1,])

full<-full0[c(rownames(nzv[nzv$percentUnique>0.1,]),cat_var0)]

#Feature engineering numeric variable
#Replace missing value with mean of numeric variable
#catogorising variables NZV
num_var<-names(full)[which(sapply(full,is.numeric))]
cat_var<-names(full)[which(sapply(full,is.factor))]
full[sapply(full,is.numeric)]<-lapply(full[sapply(full,is.numeric)],function(x) ifelse(is.na(x),mean(x,na.rm=T),x))
table(is.na(full[num_var]))

#Outlier treatment
fulltemp<-full[num_var]
for(i in 1:ncol(fulltemp)){ 
  qnt <- quantile(fulltemp[,i], probs=c(.25, .75), na.rm = T)
  caps <- quantile(fulltemp[,i], probs=c(.01, .99), na.rm = T)
  H <- 1.5 * IQR(full[,i], na.rm = T)
  fulltemp[,i][fulltemp[,i] < (qnt[1] - H)] <- caps[1]
  fulltemp[,i][fulltemp[,i] > (qnt[2] + H)] <- caps[2]
}

#Feature scaling
fulltemp=fulltemp[,-1]
fulltemp=fulltemp[,!colnames(fulltemp)%in%c('price_doc'),drop=F]
fulltemp = data.frame(scale(fulltemp))

#PCA
library(caret)
library(e1071)
#pca = preProcess(x = fulltemp, method = 'pca',pcaComp = 25)
#fulltemp$price_doc = full$price_doc
#fulltemp =predict(pca,fulltemp)
#summary(fulltemp$price_doc)

#full<-bind_cols(fulltemp,full0[cat_var])

#PCA
pca2<-prcomp(fulltemp,scale. = T)
head(pca2$scale)
biplot(pca2,scale = 0)
var<-pca2$sdev^2
var[1:10]
#proportion of variance
prop_va<-var/sum(var)
plot(prop_va)
plot(cumsum(prop_va))

fulltemp <- data.frame(price_doc=full$price_doc, pca2$x)
fulltemp <- fulltemp[,1:50]
full<-bind_cols(fulltemp,full[cat_var])

#Replace missing value with highest count of catogorical variable
colSums(is.na(full[cat_var]))

summary(full$product_type)
full$product_type[is.na(full$product_type)]='Investment'

table(is.na(full)) #Thier is no missing values in the data

#univariate analysis skip

#Data sampling
#other_var #column number are  2,13
#full=full[,-c(2,13)]
for(i in 1:ncol(full)){
  if(is.factor(full[,i])){
    (full[,i])=as.numeric(full[,i])
  }
}

full<-as.data.frame(lapply(full,as.numeric))
train2<-train1<-full[1:30471,]
test1<-full[30472:38133,]

#corelation
#a=cor(train1[num_var])
#plot(a)
# Transformed output applying BOX COX tranfermatio
#library('caret')
#c<- caret:: BoxCoxTrans(train1$price_doc)
#print(c)
#train1$price_doc=train1$price_doc^0.2

#linear regression
srb.lm<-lm(price_doc~.,data=train1)
summary(srb.lm) #R'2 is very less

#gbm
srb.gbm<-gbm(price_doc~., data=train1, distribution = 'gaussian',
             shrinkage = 0.03,
             interaction.depth = 10,
             #bag.fraction = 0.6,
             #n.minobsinnode = 1,
             cv.folds = 2,
             #keep.data = F,
             verbose = F,
             n.trees = 500)
summary(srb.gbm)


prd.gb<-predict(srb.gbm,test1,type='response',n.trees = 500)
submit<-data.frame(id=test$id,price_doc=prd.gb)
write.csv(submit,file = 'srb.submit.csv',row.names = F)

#Accuracy
prd_train<-predict(srb.gbm,train1,type = 'response',n.trees = 500)
sse1<-sum((prd_train-train1$price_doc)^2)
sst1<- sum((train1$price_doc-mean(prd_train))^2)
rsquare<- 1-(sse1/sst1)

sse<-sum((prd.gb-test1$price_doc)^2)
sst<- sum((test1$price_doc-mean(prd.gb))^2)
e=(sse/sst)
rsquare=1-e
rsquare

#xgboost
