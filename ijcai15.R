###线性回归预测和非线性回归（回归树）官网的最好成绩是0.7的auc,而我cv10 0.64+，增加了部分特征提高了
###用户成为那家店的忠实用户无非看重：品牌偏好，价格偏好(没有)，节日偏好，单品偏好（单品详细信息没有）
###
####商家来说：销量排名
#阿里ijcai15,Repeat Buyers Prediction，时间段为511-1112,未来6个月的购物是否为1.
require(data.table)
require(plyr)
require(GGally)
library(FeatureHashing)
require(pROC)
require(xgboost)
train<-fread("E:/kaggle/IJCAI15 Data/IJCAI15 Data/data_format1/train_format1.csv")
test<-fread("E:/kaggle/IJCAI15 Data/IJCAI15 Data/data_format1/test_format1.csv")
user_info<-fread("E:/kaggle/IJCAI15 Data/IJCAI15 Data/data_format1/user_info_format1.csv")
log<-fread("E:/kaggle/IJCAI15 Data/IJCAI15 Data/data_format1/user_log_format1.csv")
###user_info 存在缺失值，占比很小
user_info[is.na(user_info)]<--999
###格式二的数据集中一张宽表，特征提取比较麻烦
train<-fread("E:/kaggle/IJCAI15 Data/IJCAI15 Data/data_format2/train_format2.csv")
train[,n:=length(unlist(strsplit(activity_log,"#")))]

###基本信息，log里面需要对每个店做些统计。针对UM做汇总，还要对客户做下，不然有些用户的单店交互信息却好似很少
log$time_stamp<-as.Date(as.character(log$time_stamp+20150000),format="%Y%m%d")
log_agg<-log[,.(n=.N,lastdate=max(time_stamp),firstdate=min(time_stamp),iter=max(time_stamp)-min(time_stamp),n_0=sum(action_type==0),
       
       n_1=sum(action_type==1),n_2=sum(action_type==2),n_3=sum(action_type==3),
       n_item=length(unique(item_id)),n_cat=length(unique(cat_id)),n_brand=length(unique(brand_id)),brands=paste(brand_id,collapse = ","),
       cats=paste(cat_id,collapse = ",") ),by=c("user_id","seller_id")]

log_agg1<-log[,.(n=.N,lastdate=max(time_stamp),firstdate=min(time_stamp),iter=max(time_stamp)-min(time_stamp),n_0=sum(action_type==0),
                
                n_1=sum(action_type==1),n_2=sum(action_type==2),n_3=sum(action_type==3),
                n_item=length(unique(item_id)),n_cat=length(unique(cat_id)),n_brand=length(unique(brand_id)),brands=paste(brand_id,collapse = ","),
                cats=paste(cat_id,collapse = ",") ),by=c("user_id")]

log_agg$firstdate_flag<-ifelse(log_agg$firstdate=="2015-11-11",1,0)
names(log_agg)[2]<-"merchant_id"
log_agg$iter<-as.numeric(log_agg$iter)

log_agg1$firstdate_flag<-ifelse(log_agg1$firstdate=="2015-11-11",1,0)
log_agg1$iter<-as.numeric(log_agg1$iter)
###查看特征的相关度
ggcorr(log_agg[train,on=c("user_id","merchant_id")],label=TRUE)
##看下用户的交互变化情况,除了双11前基本都比较平稳。。。
plot(as.vector(table(log$time_stamp)))

###基本也能看出问题，过半的用户行为是从双11开始，比较后得到首次节日购物用户的转化率相对比较低
train_new<-log_agg1[user_info[log_agg[train,on=c("user_id","merchant_id")],on="user_id"],on="user_id"]
test_new<-log_agg1[user_info[log_agg[test,on=c("user_id","merchant_id")],on="user_id"],on="user_id"]

train_label<-train$label
###建模开始啦,结果表示概率都一样，样本正负样本的比例约15:1，正的样本略少。。。
### 商家基本没变化，用户基本都是新的用户。这个数据有点类似广告点击


param <- list(objective = "binary:logistic", max_depth  = 30,
              booster = "gbtree", eta = 0.0008,gamma=0.01,
              eval_metric = "auc")

###交叉验证，历史成绩0.63
train_new<-as.data.frame(train_new)
test_new<-as.data.frame(test_new)
cv.res <- xgb.cv(data = xgb.DMatrix(as.matrix(train_new[,c(2:5,8:16)]), label = train_new$label),nround=30,params =param,
                                nfold = 5)

bst<-xgb.train(data = xgb.DMatrix(as.matrix(train_new[,c(2:5,8:16)]), label = train_new$label),nround=60,params =param,
       watchlist = list(eval = xgb.DMatrix(as.matrix(train_new[,c(2:5,8:16)]), label = train_new$label), train = xgb.DMatrix(as.matrix(train_new[,c(2:5,8:16)]), label = train_new$label)))

out <- predict(bst, xgb.DMatrix(as.matrix(test_new[,c(2:5,8:16)])))

write.csv(test,"E:/kaggle/IJCAI15 Data/IJCAI15 Data/submission.csv",col.names = F,quote = F)


####hash trick 试试看
b <- 2 ^ 12+1000
f <- ~ age_range + gender +merchant_id+n+iter+n_0+n_1+n_2+n_3+n_item+n_cat+n_brand+n+iter+
  i.n_0+i.n_1+i.n_2+i.n_3+i.n_item+i.n_cat+i.n_brand+split(cats, delim = ",")+split(brands, delim = ",")+split(i.cats, delim = ",")+split(i.brands, delim = ",")-1

f <- ~ age_range + gender +merchant_id+n+iter+n_0+n_1+n_2+n_3+n_item+n_cat+n_brand+n+iter+
  split(cats, delim = ",")+split(brands, delim = ",")-1


X_train <- hashed.model.matrix(f, train_new, b)
X_test  <- hashed.model.matrix(f, test_new,  b)

cv.res <- xgb.cv(data = xgb.DMatrix(X_train, label = train_new$label),nround=20,params =param,
                 nfold = 5)
bst<-xgb.train(data = xgb.DMatrix(X_train, label = train_new$label),nround=60,params =param,
               watchlist = list(eval = xgb.DMatrix(X_train, label = train_new$label), train = xgb.DMatrix(X_train, label = train_new$label)))

out <- predict(bst, xgb.DMatrix(X_test))

write.csv(test,"E:/kaggle/IJCAI15 Data/IJCAI15 Data/submission2.csv",col.names = F,quote = F)
