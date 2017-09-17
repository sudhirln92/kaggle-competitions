# Credit card fraud detection detection
library(pastecs)#descriptive stats
library(moments)#skewness
library(fmsb)#VIF
library(caret)# confusion matrix
library(e1071) # confusion matrix
library(ROCR)# ROC curve

library(rpart) # Decision tree
library(ROSE) # Imbalanced data
library(rpart.plot) #Fancy plot
library(RColorBrewer)
library(rattle)


# Data import
setwd("/home/sudhir/R/Credit card fraud")
#card=read.csv('creditcard.csv')
card=readRDS('card.RDS')
head(card)
str(card)
card$Class=as.factor(card$Class)
#card$Time=as.numeric(card$Time)
#saveRDS(card,'card.RDS')

#Finding missing value
summary(is.na(card))
table(is.na(card))

# Data exploration
table(card$Class)

# Univariate analysis
plot(density(card$Time),xlab = 'Time',main="Credit card fraud detection",col='green')
hist(card$Time,xlab = 'Time',main="Histogram of Credit card fraud detection",col='red')
boxplot(card$Time,xlab = 'Time',main="Boxplot of Credit card fraud detection",col='yellow')
skewness(card$Time)

plot(density(card$V1),xlab = 'V1',main="Credit card fraud detection",col='red')
hist(card$V1,xlab = 'V1',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V1,xlab = 'V1',main="Boxplot of Credit card fraud detection",col='yellow')
skewness(card$V1)

plot(density(card$V2),xlab = 'V2',main="Credit card fraud detection",col='blue')
hist(card$V2,xlab = 'V2',main="Histogram of Credit card fraud detection",col='green')
boxplot(card$V2,xlab = 'V2',main="Boxplot of Credit card fraud detection",col='yellow')
skewness(card$V2)

plot(density(card$V3),xlab = 'V3',main="Credit card fraud detection",col='green')
hist(card$V3,xlab = 'V3',main="Histogram of Credit card fraud detection",col='red')
boxplot(card$V3,xlab = 'V3',main="Boxplot of Credit card fraud detection",col='blue')
skewness(card$V3)

plot(density(card$V4),xlab = 'V4',main="Credit card fraud detection",col='brown')
hist(card$V4,xlab = 'V4',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V4,xlab = 'V4',main="Boxplot of Credit card fraud detection",col='purple')
skewness(card$V4)

plot(density(card$V5),xlab = 'V5',main="Credit card fraud detection",col='red')
hist(card$V5,xlab = 'V5',main="Histogram of Credit card fraud detection",col='yellow')
boxplot(card$V5,xlab = 'V5',main="Boxplot of Credit card fraud detection",col='green')
skewness(card$V5)

plot(density(card$V6),xlab = 'V6',main="Credit card fraud detection",col='green')
hist(card$V6,xlab = 'V6',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V6,xlab = 'V6',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V6)

plot(density(card$V7),xlab = 'V7',main="Credit card fraud detection",col='green')
hist(card$V7,xlab = 'V7',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V7,xlab = 'V7',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V7)

plot(density(card$V8),xlab = 'V8',main="Credit card fraud detection",col='green')
hist(card$V8,xlab = 'V8',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V8,xlab = 'V8',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V8)

plot(density(card$V9),xlab = 'V9',main="Credit card fraud detection",col='green')
hist(card$V9,xlab = 'V9',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V9,xlab = 'V9',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V9)

plot(density(card$V10),xlab = 'V10',main="Credit card fraud detection",col='green')
hist(card$V10,xlab = 'V10',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V10,xlab = 'V10',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V10)

plot(density(card$V11),xlab = 'V11',main="Credit card fraud detection",col='green')
hist(card$V11,xlab = 'V11',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V11,xlab = 'V11',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V11)

plot(density(card$V12),xlab = 'V12',main="Credit card fraud detection",col='green')
hist(card$V12,xlab = 'V12',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V12,xlab = 'V12',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V12)

plot(density(card$V13),xlab = 'V13',main="Credit card fraud detection",col='green')
hist(card$V13,xlab = 'V13',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V13,xlab = 'V13',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V13)

plot(density(card$V14),xlab = 'V14',main="Credit card fraud detection",col='green')
hist(card$V14,xlab = 'V14',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V14,xlab = 'V14',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V14)

plot(density(card$V15),xlab = 'V15',main="Credit card fraud detection",col='green')
hist(card$V15,xlab = 'V15',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V15,xlab = 'V15',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V15)

plot(density(card$V16),xlab = 'V16',main="Credit card fraud detection",col='green')
hist(card$V16,xlab = 'V16',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V16,xlab = 'V16',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V16)

plot(density(card$V17),xlab = 'V17',main="Credit card fraud detection",col='green')
hist(card$V17,xlab = 'V17',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V17,xlab = 'V17',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V17)

plot(density(card$V18),xlab = 'V18',main="Credit card fraud detection",col='green')
hist(card$V18,xlab = 'V18',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V18,xlab = 'V18',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V18)

plot(density(card$V19),xlab = 'V19',main="Credit card fraud detection",col='green')
hist(card$V19,xlab = 'V19',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V19,xlab = 'V19',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V19)

plot(density(card$V20),xlab = 'V20',main="Credit card fraud detection",col='green')
hist(card$V20,xlab = 'V20',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V20,xlab = 'V20',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V20)

plot(density(card$V21),xlab = 'V21',main="Credit card fraud detection",col='green')
hist(card$V21,xlab = 'V21',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V21,xlab = 'V21',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V21)

plot(density(card$V22),xlab = 'V22',main="Credit card fraud detection",col='green')
hist(card$V22,xlab = 'V22',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V22,xlab = 'V22',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V22)

plot(density(card$V23),xlab = 'V23',main="Credit card fraud detection",col='green')
hist(card$V23,xlab = 'V23',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V23,xlab = 'V23',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V23)

plot(density(card$V24),xlab = 'V24',main="Credit card fraud detection",col='green')
hist(card$V24,xlab = 'V24',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V24,xlab = 'V24',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V24)

plot(density(card$V25),xlab = 'V25',main="Credit card fraud detection",col='green')
hist(card$V25,xlab = 'V25',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V25,xlab = 'V25',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V25)

plot(density(card$V26),xlab = 'V26',main="Credit card fraud detection",col='green')
hist(card$V26,xlab = 'V26',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V26,xlab = 'V26',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V26)

plot(density(card$V27),xlab = 'V27',main="Credit card fraud detection",col='green')
hist(card$V27,xlab = 'V27',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V27,xlab = 'V27',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V27)

plot(density(card$V28),xlab = 'V28',main="Credit card fraud detection",col='green')
hist(card$V28,xlab = 'V28',main="Histogram of Credit card fraud detection",col='blue')
boxplot(card$V28,xlab = 'V28',main="Boxplot of Credit card fraud detection",col='brown')
skewness(card$V28)

#Bivariate analysis
library(ggplot2)
ggplot(subset(card, Class %in% c("0","1")), aes(x=Class,y=V1,color=V2))+geom_point()

# Model building
 # Decision tree
fit1=rpart(Class~.,method = 'class',data=card)
fit1$cptable
plot(fit1)
text(fit1)

# Model validation
pred=predict(fit1,newdata=card,type='class')
confusionMatrix(pred,card$Class)

# Handling imblanced data
 # Over sampling
data_over=ovun.sample(Class~.,data=card,method = 'over')$data
table(data_over$Class)

 # under sampling
data_under=ovun.sample(Class~., data=card, method = 'under',N=984)$data
table(data_under$Class)
 
 # data balanced both
data_both=ovun.sample(Class~.,data=card,method = 'both', p=0.5, N=1200,seed = 1 )$data
table(data_both$Class)

 # data ROSE
data_rose=ROSE(Class~., data=card, seed=1)$data
table(data_rose$Class)

# build decision tree model
tree.rose=rpart(Class~., data= data_rose)
tree.over=rpart(Class~., data= data_over)
tree.under=rpart(Class~., data=data_under)
tree.both=rpart(Class~., data=data_both)

# make prediction on unseen data
pred.tree.rose=predict(tree.rose, newdata=card)
pred.tree.over=predict(tree.over, newdata=card)
pred.tree.under=predict(tree.under, newdata=card)
pred.tree.both=predict(tree.both, newdata=card)

# finding AUC: area under curve
roc.curve(card$Class, pred.tree.rose[,2])
roc.curve(card$Class,pred.tree.over[,2])
roc.curve(card$Class, pred.tree.both[,2])
roc.curve(card$Class, pred.tree.under[,2])

# Model works good for over sampling
# lets split data train and test
sp1=sample(nrow(data_over),nrow(data_over)*.7)
train=data_over[sp1,]
test=data_over[-sp1,]

 # Model building
model.dc.sp=rpart(Class~., data=train,method='class')
model.dc.sp$cptable
plot(model.dc.sp)
text(model.dc.sp)
fancyRpartPlot(model.dc.sp)

#Pruning of tree
prn.sp=prune(model.dc.sp, cp=model.dc.sp$cptable[which.min(model.dc.sp$cptable[,"xerror"]),"CP"])
fancyRpartPlot(prn.sp)

#Predicting the model
pre.sp=predict(prn.sp,newdata = test,type = 'class')
accuracy.sp=mean(pre.sp==test$Class)

confusionMatrix(pre.sp,test$Class)

pre.dp=predict(prn.sp,newdata = card,type = 'class')
table(pre.dp)

submit=data.frame(Class=pre.dp)
write.csv(submit,'submitcard.csv',row.names = F)

