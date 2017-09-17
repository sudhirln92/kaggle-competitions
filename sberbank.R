#Sberbank
library(gbm)
library(dplyr)
library(caret)

#Import data
setwd('/home/sudhir/R/Sberbank')
#train<-fread('train.csv',na.strings = 'NA',sep = ',')
#test<-fread('test.csv',na.strings = 'NA',sep = ',')

train<-read.csv('train.csv',na.strings = 'NA')
test<-read.csv('test.csv',na.strings = 'NA')
macro<-read.csv('macro.csv',na.strings = 'NA')

test$price_doc=0
#Feature engineering
full0<-bind_rows(train,test)
full0$timestamp<-as.Date(full0$timestamp)
macro$timestamp<-as.Date(macro$timestamp)
full0<-merge(full0,macro,by='timestamp')

str(full0)

#target variable
summary(full0$price_doc)
#near zero variance for numeric variable
num_var0<-names(full0)[which(sapply(full0,is.numeric))]
cat_var0<-names(full0)[which(sapply(full0,is.factor))]
nzv<-nearZeroVar(full0[num_var0],saveMetrics = TRUE)
dim(nzv[nzv$percentUnique>0.1,])

full<-full0[c(rownames(nzv[nzv$percentUnique>0.1,]),cat_var0)]

#missing values
table(is.na(full))
colSums(is.na(full))

#catogorising variables NZV
num_var<-names(full)[which(sapply(full,is.numeric))]
cat_var<-names(full)[which(sapply(full,is.factor))]

#Feature engineering numeric variable
#Replace missing value with mean of numeric variable
full[sapply(full,is.numeric)]<-lapply(full[sapply(full,is.numeric)],function(x) ifelse(is.na(x),mean(x,na.rm=T),x))
table(is.na(full[num_var]))

#Replace missing value with mean of numeric variable
colSums(is.na(full[cat_var]))

summary(full$product_type)
full$product_type[is.na(full$product_type)]='Investment'

summary(full$child_on_acc_pre_school.x)

#Ignore
fc=c('child_on_acc_pre_school','modern_education_share','old_education_build_share')
#univariate analysisskip
####
full<-full[,!names(full)%in%fc]

table(is.na(full)) #Thier is no missing values in the data
for(i in 1:255){
  if(is.factor(full[,i])){
    (full[,i])<-as.numeric(full[,i])
  }
}

full<-as.data.frame(lapply(full,as.numeric))
#Data sampling
train1<-full[1:30471,]
test1<-full[30472:38133,]

#corelation
a=cor(train1[num_var])
plot(a)

#Principle component analysis

pri.var<-prcomp(full[num_var],scale. = T)
names(pri.var)
pri.var$center

biplot(pri.var)
std<-pri.var$sdev
var<-std^2
var[1:10]
#proportion of variance to explain
pro_var<-var/sum(var)
head(pro_var)
#scree plot
plot(pro_var,xlab = 'Principal component', ylab = 'Proportion of explianed',type = 'b')
plot(cumsum(pro_var), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#linear regression
srb.lm<-lm(price_doc~.,data=train1[,2:290])
summary(srb.lm) #R'2 is very less


#gbm

srb.gbm<-gbm(price_doc~., data=train1, distribution = 'gaussian',
             shrinkage = 0.05,
             interaction.depth =5,
             bag.fraction = 0.6,
             n.minobsinnode = 1,
             #cv.folds = 2,
             #keep.data = F,
             verbose = F,
             n.trees = 1000)
summary(srb.gbm)

prd.gb<-predict(srb.gbm,test1,type='response',n.trees = 1000)
submit<-data.frame(id=test1$id,price_doc=prd.gb)
write.csv(submit,file = 'srb.submit.csv',row.names = F)

#xgboost
library(xgboost)
dtest <- xgb.DMatrix(data.matrix(test1))
dtrain <- xgb.DMatrix(data.matrix(train1), 
                      label =train1$price_doc)

xgb_params = list(booster="gbtree",
                  colsample_bytree= 0.7,
                  subsample = 0.7,
                  eta = 0.05,
                  objective= 'reg:linear',
                  max_depth= 5,
                  min_child_weight= 1,
                  eval_metric= "rmse")


gbdt = xgb.train(params = xgb_params,
                 data = dtrain,
                 nrounds = 300,
                 watchlist = list(train = dtrain),
                 print_every_n = 50)


my_preds <- predict(gbdt, dtest, reshape = TRUE)
submit<-data.frame(id=test1$id,price_doc=my_preds)
write.csv(submit,file = 'srb.submit.csv',row.names = F)
