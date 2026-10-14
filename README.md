# Türkçe Mini GPT — Sıfırdan Dil Modeli

Hazır kütüphane modeli kullanmadan, **PyTorch ile satır satır yazılmış** bir Transformer dil modeli.
Türkçe Vikipedi metniyle eğitildi; tüm deneyler Google Colab'da (T4 GPU) yapıldı ve kod yerel bilgisayarda değişiklik yapılmadan çalışacak şekilde yazıldı.

Bu repo, yapay zekâyı temelden öğrenme sürecimin kaydıdır: her deneyde önce bir tahmin yaptım, sonra ölçtüm ve sonuçları yorumladım.

---

## İçerik

| Defter | Konu |
|---|---|
| [`notebooks/adim1_pytorch_isinma.ipynb`](notebooks/adim1_pytorch_isinma.ipynb) | Tensörler, autograd, eğitim döngüsü, el yazısı rakam tanıma (MNIST), model boyutu ve öğrenme hızı deneyleri |
| [`notebooks/adim2_mini_turkce_gpt.ipynb`](notebooks/adim2_mini_turkce_gpt.ipynb) | Karakter tokenizasyonu, Bigram taban modeli, çok kafalı self-attention ile Mini GPT, metin üretimi |
| [`notebooks/adim3_bpe_tokenizer.ipynb`](notebooks/adim3_bpe_tokenizer.ipynb) | Metin normalleştirme, sıfırdan BPE tokenizer, ağırlık paylaşımı, erken durdurma, karakter başına adil karşılaştırma |

## Mini GPT mimarisi

- Karakter seviyesinde tokenizasyon (120 karakterlik sözlük)
- Karakter + konum gömme
- 4 Transformer bloğu: maskeli çok kafalı self-attention (4 kafa) + ileri beslemeli ağ, LayerNorm ve artık bağlantılar
- 256 boyutlu gömme, 128 karakterlik bağlam penceresi, **3,25 milyon parametre**
- Eğitim: AdamW, warmup + cosine decay, gradyan kırpma, karma hassasiyet (fp16), checkpoint'ten devam etme

Veri: Türkçe Vikipedi'den ~5 milyon karakter (%90 eğitim, %10 doğrulama).

**Tekrarlanabilirlik:** Rastgelelik `torch.manual_seed(1337)` ile sabitlendi. Defter baştan çalıştırıldığında eğitim, her ölçüm adımında aynı hata değerlerini birebir üretir (doğrulama hatası 1,275).

---

## Sonuçlar

### Dil modeli deneyleri

| Deney | Parametre | Eğitim verisi | Doğrulama hatası | Süre (T4) |
|---|---|---|---|---|
| Rastgele tahmin | — | — | 4,787 | — |
| Bigram | 14 bin | 4,5 M karakter | 2,718 | ~30 sn |
| **Mini GPT, 4 katman, bağlam 128** | 3,25 M | 4,5 M karakter | **1,275** | 170 sn |
| Mini GPT, 6 katman, bağlam 128 | 4,83 M | 4,5 M karakter | **1,235** | 249 sn |
| Mini GPT, 4 katman, bağlam 64 | 3,23 M | 4,5 M karakter | 1,314 | 161 sn |
| Mini GPT, 4 katman, az veri | 3,25 M | 0,45 M karakter | 1,622 (en iyi) → 1,882 (son) | 171 sn |

Perplexity karşılığı: rastgele 120 → Bigram 15 → Mini GPT **3,6** seçenek.

### Karakter tokenizer'a karşı kendi BPE tokenizer'ım

Aynı veri, aynı model boyutu (4 katman, 256 boyut), aynı dropout ve 5.000 adım; değişen tek şey tokenizer. Token modelinin hatası karakter başına çevrilerek karşılaştırıldı.

| Tokenizer | Sözlük | Karakter/token | Parametre | Karakter başına doğrulama hatası |
|---|---|---|---|---|
| Karakter | 120 | 1,00 | 3,25 M | 1,275 |
| **BPE (sıfırdan)** | **2.048** | **2,84** | **3,71 M** | **1,130** |

BPE tokenizer 5 M karakterlik metinde 15 saniyede eğitildi. Ağırlık paylaşımı sayesinde 17 kat büyüyen sözlük parametre sayısını yalnızca %14 artırdı.

### Örnek üretim (sıcaklık 0,8)

**Karakter modeli:**

> Osmanlı İmparatorluğu ve Musullah Sultan Savaşı'nı askerdin tabii sözlerle karşılaştırdı. 1914'te Türkiye'nin İç Sırp ve Soğuk Temmuz'da yardımcısı olarak kabul edildi.

**BPE modeli:**

> İstanbul ilinde yine Sivas ilinde sivil sivil ilçe, köyleri, ilçelerinde, köylerindeki köylerde yer alan Valisi, merkez ilçelerde bulunmaktadır.
>
> Van'ın merkezindeki Mevin'in doğu sınırında Sivas ili, merkezi olarak yer alır.

Karakter modeli Türkçe ekleri, özel isimlerdeki kesme işaretini, tarih biçimlerini ve Vikipedi'nin sayfa yapısını öğrendi. BPE modeli bunlara ek olarak paragraf boyunca **konu tutarlılığı** kazandı (il ve ilçe yapısı, seçimler, Osmanlı tarihi gibi). Anlam ve dünya bilgisi ise iki modelde de bu ölçekte oluşmadı.

### PyTorch ısınma deneyleri (MNIST)

| Model | Parametre | Test doğruluğu |
|---|---|---|
| Tek katman, 32 nöron | 25 bin | %94,61 |
| Tek katman, 128 nöron | 102 bin | %96,63 |
| **İki katman, 128 + 128** | **118 bin** | **%97,33** |
| Tek katman, 512 nöron | 407 bin | %97,63 |

---

## Bulgular

1. **Getiri azalarak artıyor.** MNIST'te her 4 kat büyütme hatayı yaklaşık üçte bir azalttı. Dil modelinde %49 daha büyük model (6 katman) hatayı yalnızca %3 düşürdü.
2. **Derinlik, genişlikten verimli.** İkinci gizli katman (%16 ek parametre), 4 kat genişlemenin kazancının büyük kısmını 3,5 kat daha az parametreyle sağladı.
3. **Veri, model boyutundan değerli.** Veriyi 10 kat azaltmak hatayı 0,35 kötüleştirdi; modeli büyütmek ise yalnızca 0,04 iyileştirdi.
4. **Aşırı öğrenme gözle görülür.** Az veriyle doğrulama hatası 1.500. adımda dibe vurup yükselmeye başladı; eğitim hatası 0,55'e inerken iki eğri arasındaki fark 1,33'e açıldı. Checkpoint her ölçümde üzerine yazıldığı için en iyi model kayboldu; 3. adımda bu yüzden erken durdurma eklendi.
5. **Karıştırıcı değişken yakalandı.** Bağlam penceresini 64'e indirdiğim ilk deneyde, her adımda görülen veri de yarıya inmişti (hata 1,388). Yığın boyutunu eşitleyip tekrarlayınca bağlamın gerçek etkisinin bunun yalnızca %35'i olduğu ortaya çıktı (hata 1,314). İlk sonuç etkiyi 3 kat abartıyordu.
6. **Öğrenme hızı kritik.** Doğrusal regresyonda lr = 0,1 modeli patlattı (`nan`), lr = 0,0001 hedefe ulaşamadı ve model eksik sabiti eğimi büyüterek telafi etti.
7. **Sıcaklık, tutarlılık ile yaratıcılık arasındaki dengeyi kuruyor.** 0,5'te tekrar döngüleri, 1,5'te uydurma kelimeler; en iyi sonuç 0,8'de.
8. **Ezber ile halüsinasyon arasında keskin sınır yok.** Az veriyle eğitilen model, *"Cengiz Han'ın büyük oğlu Cuci'nin oğlu Tokay Timur soyundan gelmektedir"* cümlesini üretti. Eğitim metninde bu cümle yok; kaynaktaki *"Cuci'nin küçük oğlu Tokay Timur soyundan gelmekteydi"* cümlesindeki "küçük oğlu" düşmüş, sonraki cümleden "büyük oğlu" taşınmış ve fiil zamanı en sık kalıba (*-mektedir*) çevrilmiş. Sonuç tesadüfen doğru bir bilgi içeriyor; aynı mekanizma yanlış bilgi de üretebilir.
9. **Kelime parçaları, karakterlerden daha verimli.** Kendi BPE tokenizer'ım karakter başına hatayı 1,275'ten 1,130'a indirdi (%11). Aynı 128 birimlik pencere artık ~360 karakter görüyor ve model paragraf boyunca konuyu koruyabiliyor.
10. **Ölçü birimi önemli.** Token başına hata bölümler arasında yanıltıcıydı: doğrulama metni daha zor sıkıştığı için (2,62'ye karşı 2,86 karakter/token) doğrulama token hatası eğitimden düşük göründü. Karakter başına çevirince iki bölüm doğru şekilde hizalandı.
11. **Tahminim tutmadı: erken durdurma devreye girmedi.** Token cinsinden veri 3 kat küçüldüğü için erken aşırı öğrenme bekliyordum; oysa doğrulama hatası 5.000 adım boyunca iyileşmeye devam etti (eğitim–doğrulama farkı 0,19'a açılsa da). Daha uzun eğitim hâlâ kazandırabilir.
12. **BPE'nin kör noktası: cümle başı.** Cümle ortasındaki " İmparatorluğu" tek token olurken cümle başındaki "Osmanlı" `O` + `s` + `manlı` diye parçalandı. Tokenizer önünde boşluk olan ve olmayan kelimeyi ayrı öğreniyor; boşluksuz hâl seyrek olduğu için parçalanıyor.

---

## Çalıştırma

**Colab:** Defteri açın, `Çalışma zamanı → T4 GPU` seçin ve hücreleri sırayla çalıştırın. Veri ve checkpoint'ler Google Drive'a kaydedilir.

**Yerel bilgisayar:**

```bash
git clone https://github.com/ercanaksakal/turkce-mini-gpt.git
cd turkce-mini-gpt
pip install -r requirements.txt
jupyter notebook
```

Kod Colab dışında otomatik olarak yerel `./yapay_zeka_projesi/` klasörünü kullanır. NVIDIA GPU varsa CUDA, yoksa CPU ile çalışır.

## Yol haritası

- [x] PyTorch temelleri ve MNIST
- [x] Karakter seviyesinde Mini GPT
- [x] Erken durdurma ve doğrulama hatasına göre en iyi modeli kaydetme
- [x] Kendi BPE tokenizer'ım (kelime parçaları), tırnak ve karakter normalizasyonu
- [ ] Daha uzun eğitim ve daha fazla veri (BPE modeli 5.000 adımda hâlâ iyileşiyordu)
- [ ] Daha büyük model ve yerel GPU'da eğitim
- [ ] Açık bir modelin LoRA ile ince ayarı
- [ ] MEB mevzuatı için belgeye dayalı (RAG) asistan

## Teknolojiler

Python · PyTorch · Hugging Face Datasets · Matplotlib · Google Colab
