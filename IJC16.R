####评价函数是个带约束的准确率和召回率的综合F值
####口碑用户大部分都是淘宝用户，train和test用户有小部分重合,地方基本都重合（多的应该是抽样的原因）
#####预测每个用户在xiayigeyue不同地方可能会消费的商户
#####对于新的口碑用户的冷启动问题，如何给出LBS的推荐。用户天猫的数据价值体现在哪里，因为没有地理位置的字段，但是淘宝和口碑很多商品存在一定的重合度
####特征：首先得到每个地方的所有商家对应关系ML，需要统计U,UML,UL,ML,淘宝等纬度的特征。其中UML 的数据十分的稀疏，需要根据过去一段时间来预测未来一个月的
###根据论文得到建模的过程比较复杂
####Merchant-Related Feature:Sales Time Interval Statistics,Lifecycle,Merchant Budget,Lifespan
####Merchant-Location Feature:Count Feature,Rate Feature,Rank Feature,Daily Level Feature
require(plyr)
require(data.table)
require(reshape2)
require(ggplot2)
require(lubridate)
require(xgboost)
train<-fread("E:/kaggle/ali/data sets/ijcai2016_koubei_train",header = FALSE)
names(train)<-c("User_id","Merchant_id","Location_id","Time_Stamp")
taobao<-fread("E:/kaggle/ali/data sets/ijcai2016_taobao",header = FALSE)
names(taobao)<-c("User_id","Seller_id","Item_id","Category_id","Online_Action_id","Time_Stamp")
test<-fread("E:/kaggle/ali/data sets/ijcai2016_koubei_test",header = FALSE)
names(test)<-c("User_id","Location_id")
info<-fread("E:/kaggle/ali/data sets/ijcai2016_merchant_info",header = FALSE)
names(info)<-c("Merchant_id","Budget","Location_id_list")

###我们汇总
user<-train[,.(Merchant_id=paste(unique(Merchant_id),collapse =","),Location_id=paste(unique(Location_id),collapse =","),
               n_Merchant=length(unique(Merchant_id)),n_location=length(unique(Location_id)),
               last_time=max(Time_Stamp),first_time=min(Time_Stamp),n=.N),by="User_id"]
Merchant<-train[,.(n=sum(Time_Stamp>=20151101),Location_id=paste(unique(Location_id),collapse =","),
                   last_time=max(Time_Stamp),first_time=min(Time_Stamp)),by="Merchant_id"]
###shifoucunzaishuadan
UML<-train[,.(last_time=max(Time_Stamp),first_time=min(Time_Stamp),n=.N),by=c("User_id","Merchant_id","Location_id")]

UL<-train[,.(Merchant_id=paste(unique(Merchant_id),collapse =","),n_Merchant=length(unique(Merchant_id)),last_time=max(Time_Stamp),first_time=min(Time_Stamp),n=.N),by=c("User_id","Location_id")]



taobao_user<-taobao[,.(n=.N,Seller_id=paste(unique(Seller_id),collapse =","),Category_id=paste(unique(Category_id),collapse =","),
                       m_last_time=max(Time_Stamp),m_first_time=min(Time_Stamp)),by="User_id"]
info<-info[Merchant,on="Merchant_id"]
user<-taobao_user[user,on="User_id"]


####还有些特征比较难提取，比如店门的平均在线人数
train$Time_Stamp<-as.Date(as.character(train$Time_Stamp),"%Y%m%d")
UM<-train[,.(n=.N,iter=as.numeric(max(Time_Stamp)-min(Time_Stamp)),n_days=length(unique(Time_Stamp)),iter1=as.numeric(max(Time_Stamp)-min(Time_Stamp))),by=c("User_id","Merchant_id")]
UM1<-UM[n>1][,.(Merchant_id=Merchant_id,iter=iter/n)]
UM1<-UM1[,.(iter_med=median(iter),iter_max=max(iter),iter_min=min(iter),iter_mean=mean(iter),iter_var=var(iter)),by="Merchant_id"]

M<-train[,.(n=.N,n_days=length(unique(Time_Stamp)),iter1=as.numeric(as.Date("2015-11-30")-min(Time_Stamp))),by=c("User_id","Merchant_id")]

ML<-train[,.(n=.N,n_user=length(unique(User_id)),n_days=length(unique(Time_Stamp))
             ),by=c("Merchant_id","Location_id")]
ML<-ML[,`:=`(rank=rank(-n,ties.method="first"),u_rank=rank(-n_user,ties.method="first"))]

ML<-cbind(ML,ML[,.(rank=rank(-n,ties.method="first"),u_rank=rank(-n_user,ties.method="first")),by="Location_id"])

L<-train[,.(n=.N,n_user=length(unique(User_id))),by=c("Location_id")]

ML<-L[ML,on="Location_id"]

ML<-ML[,`:=`(rate=i.n_user/n_user)]

###计算用户的消费间隔时间的统计变量,重复消费的时间间隔，和时间段内的消费行为
U<-train[,.(med=median(as.numeric(diff(sort(Time_Stamp))),na.rm = T),max=max(as.numeric(diff(sort(Time_Stamp))),na.rm = T),
            min=min(as.numeric(diff(sort(Time_Stamp))),na.rm = T),var=var(as.numeric(diff(sort(Time_Stamp))),na.rm = T)),by=c("User_id")]

UML<-train[,.(iter=as.numeric(max(Time_Stamp)-min(Time_Stamp)),n=.N),by=c("User_id","Merchant_id","Location_id")]
U_agg<-UML[n>1][,.(User_id=User_id,iter=iter/n)][,.(iter_med=median(iter),iter_max=max(iter),iter_min=min(iter),
                                                    iter_mean=mean(iter),iter_var=var(iter)),by=c("User_id")]

U<-U_agg[U,on="User_id"]
####用户淘宝的购物行为
taobao$Time_Stamp<-as.Date(as.character(taobao$Time_Stamp),"%Y%m%d")
taobao_user<-taobao[,.(n=.N,Seller_id=paste(unique(Seller_id),collapse =","),Category_id=paste(unique(Category_id),collapse =","),
                       iter=as.numeric(max(Time_Stamp)-min(Time_Stamp)),
                       n1=sum(Online_Action_id==0),rate=sum(Online_Action_id==0)/sum(Online_Action_id==1)),by="User_id"]

#####训练数据负样本选取，用户经常去的地方，但是没有消费的
train_data<-taobao_user[ML[U[UML,on="User_id"],on=c("Merchant_id","Location_id")],on="User_id"]
####需要1000w，而有交互的30w，30：1的比例
base<-unique(as.data.frame(train)[,c(1,3)])
base<-merge(unique(as.data.frame(train)[,c(2,3)]),base,all.y=T)

