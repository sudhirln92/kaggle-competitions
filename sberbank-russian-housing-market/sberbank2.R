#Sberbank
library(randomForest)
library(gbm)
library(dplyr)
library(caret)

#Import data
setwd('/home/sudhir/R/Sberbank')
train<-read.csv('train.csv')
test<-read.csv('test.csv')

test$price_doc=0
#Feature engineering
full0<-bind_rows(train,test)
str(full0)

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

full<-bind_cols(fulltemp,full[cat_var])

#Replace missing value with highest count of catogorical variable
colSums(is.na(full[cat_var]))

summary(full$product_type)
full$product_type[is.na(full$product_type)]='Investment'

table(is.na(full)) #Thier is no missing values in the data

#univariate analysisskip

#Data sampling
#other_var #column number are  2,13
#full=full[,-c(2,13)]
for(i in 1:228){
  if(is.factor(full[,i])){
    (full[,i])=as.numeric(full[,i])
  }
}

full<-as.data.frame(lapply(full,as.numeric))
train1<-full[1:30471,]
test1<-full[30472:38133,]

#corelation
#a=cor(train1[num_var])
#plot(a)

#linear regression
srb.lm<-lm(price_doc~.,data=train1[,2:228])
summary(srb.lm) #R'2 is very less

#gbm
srb.gbm<-gbm(price_doc~., data=train1[,2:228], distribution = 'gaussian',
             shrinkage = 0.03,
             interaction.depth = 10,
             #bag.fraction = 0.6,
             #n.minobsinnode = 1,
             cv.folds = 2,
             #keep.data = F,
             verbose = F,
             n.trees = 500)
#summary(srb.gbm)


prd.gb<-predict(srb.gbm,test1,type='response',n.trees = 500)
acc<-mean((prd.gb-test$price_doc)^2)
submit<-data.frame(id=test1$id,price_doc=prd.gb)
write.csv(submit,file = 'srb.submit.csv',row.names = F)

#xgboost
