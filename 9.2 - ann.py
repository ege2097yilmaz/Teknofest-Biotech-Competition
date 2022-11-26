# -*- coding: utf-8 -*-


"""
*********************************************************************
*
*   Yazılım geliştirme hakları 
*
*  Copyright (c) 2021, NCT Akademi.
*  Bütün haklar bu şirket ve oluşturanlara sahiptir.
*
*  belirtilen şartlar ve bu kodlara ait değiştirmeler müsade edilmektedir
*  Aşağıdaki şartlar doğrultusunda açık kaynak kodları doğrudan kullanılabilir.
*  
*   * Kullanılacak olan kodlar yukarıdaki bilgilendirme ve yazılım mimarisini oluşturan 
*   kişilerin iletişim adresleri bulunmalıdır.
*   
*   * Kullanıan kütüphanelerin ve modüllerin (tensorflow, keras, pandas, numpy, matplotlib)
*   yüklenmesi ve kullanılması gerekmektedir.
*   Bunları deneyin;
*   pip install tensorflow
*   pip install numpy
*   pip install pandas
*   pip install matplotlib
*   pip install -U scikit-learn
*   pip install keras
*
*   * Kayıtlı olan excel verileri ve ana eğitim seti ile aynı dosya içerisinde çalıştı-
*   rılmalı ya da tam yol tanımlanmalıdır.
*
*   * NCT akademi şirketine ait olan lisansların belirtilmesi zorunlu değildir. Acncak
*   belirtilmesi durumunda kaynak kodlara erişim ve geliştirmede kolaylık sağlanmış olacaktır.
*
*  Bu yazılım herhangi bir zaruriyet doğrultmadan yapanlar tarafından açık kaynak olarak
*  sunulmaktadır. belirtilen 3 madde haricinde zaruriyet bulundurmamaktadır. kod üzerinde
*  değişiklikler ve iyileştirmeler serbest olarak tutulmuştur. Qt arayüzü ile ilgili 
*  herhangi bir sınırlandırma da mevcut değildir.
*
*  Bu yazılım hastaya sorulacak olan sorulardan bağlanan çıktılar ile daha önce eğitilen 
*  data snıfları üzerinden tahmin ve sınıflandırma yapmaktadır. Filtreler toplamda 5 
*  gizli tabakadan olulmaktadır. 17 adet girdi sağlanmalıdır. 17 içerisinde 2 adet kate-
*  gorik sınıf bulunmaktır. bunlara oneHotEncoding yöntemi uygulanmıştır. Çıktı olarak
*  toplamda 8 adet tanı koyma çıktısına sahiptir.
*   
*    Oluşturulma Tarihi Perşembe Mayıs 27 
*    16:19:10 2021baca
*    
*    Oluşturan: İsmail Ovalı
*    mail: ovali.ismail@gmail.com
*        
*    Oluşturan: Ege Yılmaz
*    mail: yilmazege97@gmail.com
*******************************************************************
"""


#1.kutuphaneler
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#2.veri onisleme
#veri yukleme
veriler = pd.read_excel('param.xlsx', engine='openpyxl')


x1 = veriler.iloc[2:314,1:5].values #bağımsız değişkenler (belirtiler)
x2 = veriler.iloc[2:314,6:17].values #bağımsız değişkenler (belirtiler)
x3 = veriler.iloc[2:314, 5].values #bağımsız değişkenler (belirtiler)
x4 = veriler.iloc[2:314, 17].values #bağımsız değişkenler (belirtiler)

hastalik = veriler.iloc[2:314,18].values #bağımlı değişken (hastalık)


x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
y = le.fit_transform(hastalik)


# 0:Bahar nezlesi, 1:Bel fıtığı, 2:Boyun fıtığı, 3:Enfeksiyon 
# 4:Grip , 5: Kanser , 6:Korona , 7:Yüksek kilo
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


sonveriler = pd.concat([x1, x3] , axis=1)
sonveriler = pd.concat([sonveriler, x2] , axis=1)
x = pd.concat([sonveriler, x4] ,axis=1)

# verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.4, random_state=0)

# #verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


classifier = Sequential()

# Yapay Sinir ağı
classifier.add(Dense(30, activation = 'relu' , input_dim = 26))
classifier.add(Dropout(0.25))

# classifier.add(Dense(18,  activation = 'relu'))
# classifier.add(Dropout(0.2))

classifier.add(Dense(20,kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                     activation = 'relu'))
classifier.add(Dropout(0.3))
# regularization ile ilgili kaynak https://www.machinecurve.com/index.php/2020/01/23/how-to-use-l1-l2-and-elastic-net-regularization-with-keras/

# classifier.add(Dense(10, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
#                      activation = 'relu'))
# classifier.add(Dropout(0.28))

# classifier.add(Dense(12,kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
#                      activation = 'relu'))
# classifier.add(Dropout(0.25))

# classifier.add(Dense(10,kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
#                      activation = 'relu'))
# classifier.add(Dropout(0.25))

classifier.add(Dense(8,  activation = 'softmax'))

# loss fonksiyonlarına göre aşağıdaki compiler seçimi
classifier.compile(optimizer = "adam", loss = 'binary_crossentropy',
                    metrics = ['accuracy'] )
# classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'] )


classifier.fit(X_train, y_train, verbose=1, batch_size=5, steps_per_epoch=30,
               epochs=350, shuffle = 'true', validation_data=(X_test,y_test))
# epoch sayısı batch sayısı ve steps_per_epoch sayısı ile ilgili kaynak https://stackoverflow.com/questions/59864408/tensorflowyour-input-ran-out-of-data
# https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class
# https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class

modelkaybi = pd.DataFrame(classifier.history.history)
modelkaybi.plot()
plt.show()

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)


classifier.summary()

from sklearn.metrics import multilabel_confusion_matrix
cm = multilabel_confusion_matrix(y_test,y_pred)

print('cm')
print(cm)

import tensorflow as tf
from tensorflow.keras.models import load_model

# tf.keras.models.save_model(
#     classifier,'C:/Users/EGE/Desktop/yüksek lisans/teknofest/biyoteknoloji/model' , overwrite=True, include_optimizer=True, save_format=None,
#     signatures=None, options=None, save_traces=True
# )

classifier.save('C:/Users/EGE/Desktop/yüksek lisans/teknofest/biyoteknoloji/modeller/model15.h5',
                overwrite=True, include_optimizer=True)

# classifier.save('model.h5', overwrite=True, include_optimizer=True)

# https://keras.io/api/models/model_saving_apis/

# Tahmin girdileri
Yas = 45
Burun_akintisi = 1
Ates = 38.5
Agri_siddeti = 2

# 0:Bacak, 1:Bel , 2:Genel , 3:kafa , 4:Sırt , 5:yok 
# Agri_bölgesi = [0,0,1,0,0,0]

oksuruk = 2
Geniz_akin = 1
Goz_kizarikl = 0
Bilinç_kaybi = 0
Kolestrol = 0
seker = 0
Tat_Ve_koku_kaybi = 0
Sigara = 0
Cinsiyet = 0
Alkol = 0
Kitle = 25.0

# 0: emekli, 1:hizmet, 2:tarım 3:öğrenci, 4: işçi
# Sektor = [0,0,0,1,0]

# 0:Bahar nezlesi, 1:Bel fıtığı, 2:Kanser başlangıcı, 3:Enfeksiyon 
# 4:Grip , 5: Boyun fıtığı , 6:Korona , 7:Yüksek kilo


output = classifier.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
                    0,0,1,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
                    Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
                    Alkol, Kitle, 0,0,0,0,1]])

# Tahminler için aşağıdaki kodları giricez
print('tahmin')
# print(classifier.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     0,0,0,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,1,0,0]]))

print('0:Bahar nezlesi, 1:Bel fıtığı, 2:Boyun fıtığı, 3:Enfeksiyon, 4:Grip , 5: Stres  , 6:Korona , 7:Yüksek kilo')
print(output)

output_pd = pd.DataFrame(output)
print(output_pd)

# y_proba = classifier.predict_proba([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
#                     1,0,0,0,0,0, oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
#                     Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
#                     Alkol, Kitle, 0,0,0,0,1]])
# print(y_proba) # hastalıklar için olasılık değerlerini verir.
