library(xgboost)
dtest <- xgb.DMatrix(data.matrix(test))
dtrain <- xgb.DMatrix(data.matrix(train), 
                      label =train$price_doc)

xgb_params = list(booster="gbtree",
                  #colsample_bytree= 0.7,
                  #subsample = 0.7,
                  eta = 0.05,
                  objective= 'reg:linear',
                  max_depth= 5,
                  min_child_weight= 1,
                  eval_metric= "rmse")


gbdt = xgb.train(params = xgb_params,
                 data = dtrain,
                 nrounds = 500,
                 watchlist = list(train = dtrain),
                 print_every_n = 50)


my_preds <- predict(gbdt, dtest, reshape = TRUE)
submit<-data.frame(id=test$id,price_doc=my_preds)
write.csv(submit,file = 'srb.submit.csv',row.names = F)



#3
train<-as.data.frame(lapply(train,as.numeric))
test<-as.data.frame(lapply(test,as.numeric))
