# -*- coding: utf-8 -*-
"""
11/05/2021

@author: Ege Yılmaz
"""


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#2 veri yukleme
veriler = pd.read_excel('param.xlsx')


#2.veri onisleme
x1 = veriler.iloc[2:314,1:5].values #bağımsız değişkenler (belirtiler)
x2 = veriler.iloc[2:314,6:17].values #bağımsız değişkenler (belirtiler)
x3 = veriler.iloc[2:314, 5].values #bağımsız değişkenler (belirtiler)
x4 = veriler.iloc[2:314, 17].values #bağımsız değişkenler (belirtiler)

hastalik = veriler.iloc[2:314,18].values #bağımlı değişken (hastalık)


x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)

#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
y = le.fit_transform(hastalik)


# 0:Bahar nezlesi, 1:Bel fıtığı, 2:Kanser başlangıcı, 3:Enfeksiyon 
# 4:Grip , 5: Boyun fıtığı , 6:Korona , 7:Yüksek kilo
y = y.reshape(-1,1)
ohe = preprocessing.OneHotEncoder()
y = ohe.fit_transform(y).toarray()
y = pd.DataFrame(y)

# 0:Bacak, 1:Bel , 2:Genel , 3:kafa , 4:Sırt , 5:yok 
x3 = x3.reshape(-1,1)
ohe = preprocessing.OneHotEncoder()
x3 = ohe.fit_transform(x3).toarray()
x3 = pd.DataFrame(x3)


# 0: emekli, 1:hizmet, 2:tarım 3:öğrenci, 4: işçi
x4 = x4.reshape(-1,1)
ohe = preprocessing.OneHotEncoder()
x4 = ohe.fit_transform(x4).toarray()
x4 = pd.DataFrame(x4)

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

# ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
#                         remainder="passthrough"
#                         )
# y = ohe.fit_transform(y)

sonveriler = pd.concat([x1, x3] , axis=1)
sonveriler = pd.concat([sonveriler, x2] , axis=1)
x = pd.concat([sonveriler, x4] ,axis=1)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.42, random_state=0)

# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# # logistic reg
# from sklearn.linear_model import LogisticRegression
# logr = LogisticRegression(random_state=0)
# logr.fit(X_train,y_train)

# y_pred = logr.predict(X_test)
# # print(y_pred)
# # print(y_test)

# from sklearn.metrics import confusion_matrix
# cm = multilabel_confusion_matrix(y_test,y_pred)
# print("logistic")
# print(cm)


# yakınkomşular algoritması
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7, weights = 'distance', metric='minkowski', p = 2)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import multilabel_confusion_matrix
cm1 = multilabel_confusion_matrix(y_test,y_pred)
print("KNN")
print(cm1)


# #destekvektör
# from sklearn.svm import SVC
# svc = SVC(kernel='rbf')
# svc.fit(X_train,y_train)

# y_pred = svc.predict(X_test)

# from sklearn.metrics import multilabel_confusion_matrix
# cm2 = multilabel_confusion_matrix(y_test,y_pred)
# print('SVC')
# print(cm2)


# #bayes gaussian
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)

# y_pred = gnb.predict(X_test)

# from sklearn.metrics import confusion_matrix
# cm2 = multilabel_confusion_matrix(y_test,y_pred)
# # cm3 = confusion_matrix(y_test,y_pred)
# print('GNB')
# print(cm2)

# #karar ağacı
# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier(criterion = 'entropy')

# dtc.fit(X_train,y_train)
# y_pred = dtc.predict(X_test)

# from sklearn.metrics import confusion_matrix
# cm3 = multilabel_confusion_matrix(y_test,y_pred)
# print('DTC')
# print(cm3)

# #rassal ağaçlar
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=1, criterion = 'entropy')
# rfc.fit(X_train,y_train)

# y_pred = rfc.predict(X_test)
# # y_proba = rfc.predict_proba(X_test) #olasılıkları belirlemek için kullanılır. notlarda sayfa 60ta
# # print(y_proba)

# from sklearn.metrics import multilabel_confusion_matrix
# cm4 = multilabel_confusion_matrix(y_test,y_pred)
# print('RFC')
# print(cm4)

# Tahmin girdileri

Yas = 22
Burun_akintisi = 0
Ates = 36.1
Agri_siddeti = 5

# 0:Bacak, 1:Bel , 2:Genel , 3:kafa , 4:Sırt , 5:yok 
# Agri_bölgesi = [0,0,1,0,0,0]

oksuruk = 0
Geniz_akin = 0
Goz_kizarikl = 0
Bilinç_kaybi = 0
Kolestrol = 0
seker = 0
Tat_Ve_koku_kaybi = 0 
Sigara = 0 
Cinsiyet = 0
Alkol = 0
Kitle = 23.0

# 0: emekli, 1:hizmet, 2:tarım 3:öğrenci, 4: işçi
# Sektor = [0,0,0,1,0]

# 0:Bahar nezlesi, 1:Bel fıtığı, 2:Boyun fıtığı, 3:Enfeksiyon 
# 4:Grip , 5: Strese bağlı hastalık , 6:Korona , 7:Yüksek kilo

# # Tahminler için aşağıdaki kodları giricez
print("0:Bahar nezlesi, 1:Fıtık Başlangıcı, 2: Kas ve eklem ağrıları, 3:Enfeksiyon, 4:Grip , 5: Strese bağlı hastalık , 6:Korona , 7:Yüksek kilo")
print('tahmin')
# print(knn.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
                    # 0,1,0,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
                    # Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
                    # Alkol, Kitle, 0,0,0,1,0]]))

y_proba = knn.predict_proba([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
                    0,1,0,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
                    Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
                    Alkol, Kitle, 0,0,0,1,0]])

print(y_proba) # hastalıklar için olasılık değerlerini verir.

# Output = knn.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     0,0,1,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,0,1,0]])
# print(type(Output))

# sonuçları terminale yazdırma kısmı
# if (output[0]==1)
#     print("bahar nezlesi")
    
#     else if (rfc.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     0,0,1,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,0,1,0]])[1] == 1)
#     print("Bel fıtığı başlangıcı")
    
#     else if (rfc.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     0,0,1,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,0,1,0]])[3] == 1)
#     print("Kanser başlangıcı")
    
#     else if (rfc.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     0,0,1,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,0,1,0]])[3] == 1)
#     print("Enfeksiyon")
    
#     else if (rfc.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     0,0,1,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,0,1,0]])[4] == 1)
#     print("Grip")
    
#     else if (rfc.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     0,0,1,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,0,1,0]])[5] == 1)
#     print("")
    
#     else if (rfc.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     0,0,1,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,0,1,0]])[6] == 1)
#     print("")
    
#     else if (rfc.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     0,0,1,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,0,1,0]])[6] == 1)
#     print("")