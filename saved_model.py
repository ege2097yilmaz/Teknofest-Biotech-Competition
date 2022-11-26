# -*- coding: utf-8 -*-

"""
*********************************************************************
*
*   Yazılım geliştirme hakları 
*
*   Copyright (c) 2021, NCT Akademi.
*   Bütün haklar bu şirket ve oluşturanlara sahiptir.
*
*   belirtilen şartlar ve bu kodlara ait değiştirmeler müsade edilmektedir
*   Aşağıdaki şartlar doğrultusunda açık kaynak kodları doğrudan kullanılabilir.
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
*    16:19:10 2021
*    
*    Oluşturan: İsmail Ovalı
*    mail: ovali.ismail@gmail.com
*        
*    Oluşturan: Ege Yılmaz
*    mail: yilmazege97@gmail.com
*******************************************************************
"""


# importing general moduls
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# import keras
from tensorflow.keras.models import load_model

# importing tensorflow
# import tensorflow as tf 

#  kerastan modeli yükleme
model = load_model('C:/Users/EGE/Desktop/yüksek lisans/teknofest/biyoteknoloji/modeller/model3.h5')
# https://keras.io/api/models/model_saving_apis/

# Parmetrelerin tanımlanması
Agri_bölgesi = ""
Sektor = ""
# ağrı bölgesi parametreleri
a = 0
b = 0
c = 0
d = 0
e = 0
g = 0

#sektör parametreleri
el = 0
f = 0
h = 0
j = 0
k = 0

# ************** Tahmin girdileri ****************
# 0:Bacak, 1:Bel , 2:Genel , 3:kafa , 4:Sırt , 5:yok 
print("lütfen ağrı bölgeinizi girini")
print("bacak, bel , genel , baş , sirt , yok")

girdi1 = str(input(Agri_bölgesi))
if girdi1 == "yok":
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    g = 1
    
elif girdi1 == "bacak":
    a = 1
    b = 0
    c = 0
    d = 0
    e = 0
    g = 0
    
elif girdi1 == "bel":
    a = 0
    b = 1
    c = 0
    d = 0
    e = 0
    g = 0
    
elif girdi1 == "genel":
    a = 0
    b = 0
    c = 1
    d = 0
    e = 0
    g = 0
    
elif girdi1 == "baş":
    a = 0
    b = 0
    c = 0
    d = 1
    e = 0
    g = 0
    
elif girdi1 == "sirt":
    a = 0
    b = 0
    c = 0
    d = 0
    e = 1
    g = 0
    
else:
    print("bölge yanlış girildi")


# Tahmin girdileri
Yas = 55
Burun_akintisi = 0
Ates = 36.5
Agri_siddeti = 4

# 0:Bacak, 1:Bel , 2:Genel , 3:kafa , 4:Sırt , 5:yok 
# Agri_bölgesi = [0,0,1,0,0,0]

oksuruk = 1
Geniz_akin = 0
Goz_kizarikl = 0
Bilinç_kaybi = 1
Kolestrol = 1
seker = 1
Tat_Ve_koku_kaybi = 0
Sigara = 1
Cinsiyet = 0
Alkol = 1
Kitle = 24.0

# 0: emekli, 1:hizmet, 2:tarım 3:öğrenci, 4: işçi
print("lütfen çalıştığınız sektörü giriniz")
print("Sektörler:  emekli, hizmet, tarım, öğrenci, işçi")

girdi2 = str(input(Sektor))
if girdi2 == "emekli":
    el = 1
    f = 0
    h = 0
    j = 0
    k = 0
    
elif girdi2 == "hizmet":
    el = 0
    f = 1
    h = 0
    j = 0
    k = 0

elif girdi2 == "tarım":
    el = 0
    f = 0
    h = 1
    j = 0
    k = 0
  
elif girdi2 == "öğrenci":
    el = 0
    f = 0
    h = 0
    j = 1
    k = 0
    
elif girdi2 == "işçi":
    el = 0
    f = 0
    h = 0
    j = 0
    k = 1

else:
    print("sektör yanlış girildi")

# 0:Bahar nezlesi, 1:Bel fıtığı, 2:Kanser başlangıcı, 3:Enfeksiyon 
# 4:Grip , 5: Boyun fıtığı , 6:Korona , 7:Yüksek kilo

# tahmin metodu
output = model.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
                    a,b,c,d,e,g , oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
                    Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
                    Alkol, Kitle, el,f,h,j,k]])

# tahmin ve hastalıkların yazdırılması
print("""hastalıklar: 
      ... 0:Bahar nezlesi, 1:Fıtık Başlangıcı, 2:Eklem ve Kas ağrıları, 3:Enfeksiyon, 
      ... 4:Grip , 5: Kanser Türevi hastalıklar , 6:Korona , 7:Yüksek kilo""")
print(output)

# ön işlemler için datafrme dönüşümü
output_df = pd.DataFrame(output)

# görselleştirme
# plt.table(rowLabels=output,
          # colLabels=...)