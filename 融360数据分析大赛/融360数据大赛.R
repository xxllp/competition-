###这次比赛需要预测二次成交
require(data.table)
require(ggplot2)
require(xgboost)
require(caret)
train<-fread("E:/kaggle/dataseriesv2/初赛数据/data/train.txt")
test<-fread("E:/kaggle/dataseriesv2/初赛数据/data/test.txt")
user_info<-fread("E:/kaggle/dataseriesv2/初赛数据/data/user_info.txt")
consumption_recode<-fread("E:/kaggle/dataseriesv2/初赛数据/data/consumption_recode.txt")

train<-merge(train,user_info,by="user_id",all.x = TRUE)
test<-merge(test,user_info,by="user_id",all.x = TRUE)

train$tm_encode<-NULL
train<-train[!duplicated(train),]
test<-test[!duplicated(test),]


###数据预处理，包含缺失值处理。

train$user_id<-NULL
train[is.na(train)]<--999
train$age<-as.numeric(train$age)
train$age<-ifelse(is.na(train$age),28,train$age)

testid<-test$user_id
test$user_id<-NULL
test[is.na(test)]<--999
test$age<-as.numeric(test$age)
test$age<-ifelse(is.na(test$age),28,test$age)

traindata<-createDataPartition(y=train$lable,p=0.75,list=F)
training=train[traindata,]

testing=train[-traindata,]

rfmodel <- train(lable~.,data=train,
                 method = "rf",metric = "ROC",
                 tuneGrid=expand.grid(mtry=10),
                 trControl = trainControl(method = "repeatedcv",
                                          repeats = 5,classProbs = TRUE,summaryFunction = twoClassSummary))





gbmmodel <- train(lable~.,data=train,
                  method = "gbm",metric = "ROC",
                  tuneGrid=expand.grid(n.trees=50,interaction.depth = 10,shrinkage=0.1,n.minobsinnode=1),
                  trControl = trainControl(method = "repeatedcv",
                                           repeats = 5,classProbs = TRUE,summaryFunction = twoClassSummary
                                           ))


###建模（模型和变量选择，参数调优 ），最后的得分大概是0.54。比较的低~，准确率刚过50%
xgtmodel <- train(lable~.,data=train,
                  method = "xgbTree",metric = "ROC",
                  tuneGrid=expand.grid(nrounds=100,
                                       max_depth  = 12,
                                       eta=0.01,
                                       gamma=1,
                                       colsample_bytree=1,
                                       min_child_weight=1),
                  trControl = trainControl(method = "repeatedcv",repeats = 3,
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary))
###输出结果
target<-predict(xgtmodel,newdata=test,type = "prob")[,1]
submission=data.frame(user_id=testid,probability=target)
write.csv(submission,"E:/kaggle/dataseriesv2/初赛数据/data/submission.txt",row.names=F,quote=F)
