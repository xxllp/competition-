#coding=utf-8
###jiaba 分词得到的不少专业词还需要进一步细化，不然训练不出来效果；比如网络工程，需要拆为网络/工程。效果改善不是很明显.目前最好的成绩在48%，主要是职位和size准确度还是比较低，
###考虑添加特征：进一步需要考虑职位的相似度，结合行业。简单的文本处理还是没啥用的
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2,SelectPercentile
from sklearn.preprocessing import Binarizer
data=pd.read_csv(ur'd:/download/data.csv',encoding='GBK')
data.head()
#data=data[data['major'].isnull()==False]
data['major']=data['major'].fillna('')
data['industry1']=data['industry1'].fillna('')
data['industry3']=data['industry3'].fillna('')
from jieba import Tokenizer
cut=Tokenizer().lcut

TfidfVectorizer=CountVectorizer
def tokenizer(contents):
   a=[e for e in cut(contents,cut_all=True) if len(e)<=3]
   return a

def pre(contents):
    a=re.sub('')
    return a
tv = TfidfVectorizer(max_features=300,tokenizer=tokenizer )
tfidf_train = tv.fit_transform(data.major)
print 'step1'
major_tdidf=tv.transform(data.major)
print 'step1'
print tv.get_feature_names()

industry1=data.industry1.value_counts().reset_index().head(100)['index']
#industry2=data.industry2.value_counts().reset_index().head(100)['index'].append(test.industry2.value_counts().reset_index().head(100)['index'])
industry3=data.industry3.value_counts().reset_index().head(100)['index']
position_name1=data.position_name1.value_counts().reset_index().head(100)['index']
position_name2=data.position_name2.value_counts().reset_index().head(100)['index']
position_name3=data.position_name3.value_counts().reset_index().head(100)['index']
print 'step1'
#data['industry1']=data.industry1.apply( lambda e : e if e in list(industry1) else 'other')
#data['industry2']=data.industry2.apply( lambda e : e if e in list(industry2) else 'other')
#data['industry3']=data.industry3.apply( lambda e : e if e in list(industry3) else 'other')
tv = TfidfVectorizer(max_features=300,tokenizer=tokenizer)
tfidf_train = tv.fit_transform(data.industry1)
industry1_tdidf=tv.transform(data.industry1)

industry3_tdidf=tv.transform(data.industry3)
#data['position_name1']=data.position_name1.apply( lambda e : e if e in list(position_name1) else 'other')
tv = TfidfVectorizer(max_features=300,tokenizer=tokenizer)
tfidf_train = tv.fit_transform(data.position_name1)
position1_tdidf=tv.transform(data.position_name1)
data['position_name2']=data.position_name2.apply( lambda e : e if e in list(position_name2) else 'other')
#data['position_name3']=data.position_name3.apply( lambda e : e if e in list(position_name3) else 'other')
position3_tdidf=tv.transform(data.position_name3)
print 'step1'
data.year1=data.year1.apply(lambda e : int((e/365.0)))
#data.year2=data.year2.apply(lambda e : int((e/365.0)))
data.year3=data.year3.apply(lambda e : int((e/365.0)))
print 'step1'
data.age=data.age.apply(lambda e: int(re.findall('\d+',e)[0]) if len(re.findall('\d+',e))>0 else 0)
data.age=data.age.apply(lambda e: e if e >=18 else 18)
def pre_age(age):
    if age==18 :
        return 0
    elif age <=30 :
        return 1
    elif age <=40:
        return 2
    elif age <=60:
        return  3
    else:
        return 4
data.age=data.age.apply(lambda e: pre_age(e))
#binarizer = Binarizer().fit(data.age)
#data.age=binarizer.transform(data.age)
data.gender=data.gender.apply(lambda e: 1 if e in (u'男','Male')else 0)
print 'step1'
tmp=pd.concat([pd.get_dummies(data.gender, prefix='gender'),pd.DataFrame(major_tdidf.toarray()),
               pd.get_dummies(data.size1, prefix='size1'),pd.get_dummies(data.size3,prefix='size3'),pd.get_dummies(data.salary1,prefix='salary1'),pd.get_dummies(data.salary3,prefix='salary3'),
               pd.DataFrame(industry1_tdidf.toarray()),pd.DataFrame(industry3_tdidf.toarray()),pd.DataFrame(position1_tdidf.toarray()),pd.DataFrame(position3_tdidf.toarray()),
               pd.get_dummies(data.year1,prefix='year1'),pd.get_dummies(data.year3,prefix='year3'),pd.get_dummies(data.age, prefix='age')], axis=1)
print 'step1'
###变量筛选，变量太多了，超1000多了
ch2 = SelectKBest(chi2, k=400)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals.joblib import Parallel, delayed

X_train, X_test, y_train, y_test = train_test_split(tmp, data['degree'], test_size=0.2, random_state=42)
clf=RandomForestClassifier(max_depth=5, n_estimators=500)
clf=SVC(C=0.1)
clf=LogisticRegression(C=1, penalty='l2', tol=1e-6)
scores = cross_val_score(clf, tmp, data['degree'], cv=5)
print scores
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


tmp=pd.concat([tmp,pd.get_dummies(pd.DataFrame(clf.predict(tmp),columns=['degree']),prefix='degree')], axis=1)
#print clf.coef_
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
degree_loss= (y_pred==y_test)*0.35

X_train, X_test, y_train, y_test = train_test_split(tmp, data['salary2'], test_size=0.2, random_state=42)
#X_train = ch2.fit_transform(X_train, y_train)
clf=RandomForestClassifier(max_depth=5, n_estimators=500)
clf=SVC(C=0.1)
clf=LogisticRegression(C=1, penalty='l1', tol=1e-6)

scores = cross_val_score(clf, tmp, data['salary2'], cv=5)
print scores
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

tmp=pd.concat([tmp,pd.get_dummies(pd.DataFrame(clf.predict(tmp),columns=['salary2']),prefix='salary2')], axis=1)
print clf.coef_
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
salary_loss=(y_pred==y_test)*0.86

X_train, X_test, y_train, y_test = train_test_split(tmp, data['size2'], test_size=0.2, random_state=42)
clf=RandomForestClassifier(max_depth=5, n_estimators=500)
clf=SVC(C=0.1)
clf=LogisticRegression(C=1, penalty='l1', tol=1e-6)
scores = cross_val_score(clf, tmp, data['size2'], cv=5)
print scores

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

tmp=pd.concat([tmp,pd.get_dummies(pd.DataFrame(clf.predict(tmp),columns=['size2']),prefix='size2')], axis=1)
#print clf.coef_
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
size_loss= (y_pred==y_test)*0.74

X_train, X_test, y_train, y_test = train_test_split(tmp, data['position_name2'], test_size=0.2, random_state=42)
clf=RandomForestClassifier(max_depth=5, n_estimators=500)
clf=SVC(C=0.1)
clf=LogisticRegression(C=1, penalty='l1', tol=1e-6)
scores = cross_val_score(clf, tmp, data['position_name2'], cv=5)
print scores

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
#print clf.coef_
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
position_loss=(y_pred==y_test)*2.25

loss=(degree_loss+salary_loss+size_loss+position_loss)/(0.35+0.86+0.74+2.25)
print loss.sum()/len(loss)
