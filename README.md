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
| [`notebooks/adim4_acik_model_kiyas.ipynb`](notebooks/adim4_acik_model_kiyas.ipynb) | Kendi modelim ile Qwen3 (0.6B, 1.7B) ve Kumru-2B'nin aynı ölçüyle karşılaştırması, tokenizer analizi, sızıntı testi |

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

### Hazır açık modellerle aynı ölçüde karşılaştırma

Aynı 461 bin karakterlik Vikipedi doğrulama metni ve aynı ölçü (karakter başına hata). "Kendi metnim" sütunu, hiçbir modelin eğitimde görmediği, internette yayımlanmamış bir okul metni (sızıntı testi). Ölçümler Colab L4 GPU üzerinde, float32 ile yapıldı.

| Model | Parametre | Türkçeye özel | Karakter/token | Vikipedi | Kendi metnim | Fark |
|---|---|---|---|---|---|---|
| Bizim BPE Mini GPT | 3,7 M | Evet | 2,62 | 1,128 | 1,428 | +0,300 |
| Qwen3-0.6B-Base | 0,60 B | Hayır | 2,45 | 0,965 | 0,923 | −0,042 |
| Qwen3-1.7B-Base | 1,72 B | Hayır | 2,45 | 0,780 | 0,828 | +0,048 |
| **Kumru-2B-Base** | **2,38 B** | **Evet** | **3,11** | **0,611** | **0,688** | +0,077 |

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
13. **Ölçüm hatası, sonucu tersine çevirebilir.** Kumru-2B ilk ölçümde 2,277 çıktı; 2 milyar parametrelik bir modelin bizim modelden kötü olması mümkün değildi. Sebep: model her girdinin başında `<BOS>` tokenı bekliyor (BOS'suz token başına hata 15,9, BOS'lu 1,5). Düzeltmeden sonra 0,611 oldu. BOS kullanmayan Qwen'in skoru değişmedi; bu, düzeltmenin doğruluğunun kontrolüydü.
14. **Küçük test parçası yanıltır.** Doğrulama metninin ilk 100 bin karakteri, metnin tamamından 0,08 daha zor çıktı (1,209'a karşı 1,128). Bu yüzden tüm karşılaştırmalar metnin tamamında yapıldı.
15. **Boyut mu, dil mi? Yine bir karıştırıcı değişken.** Kumru hem Türkçeye özel hem Qwen-0.6B'den 4 kat büyük. Ayırmak için Kumru'ya yakın boyutta Qwen3-1.7B eklendi. Qwen ailesinde 2,9 kat büyüme hatayı 0,185 düşürdü; bu eğilim logaritmik varsayılırsa, Kumru ile Qwen-1.7B arasındaki 1,4 katlık boyut farkı ~0,06 açıklıyor. Kalan ~0,11 büyük olasılıkla Türkçeye özel eğitimden geliyor. Kabaca: Kumru'nun üstünlüğünün **üçte ikisi dilden, üçte biri boyuttan**. (Varsayımlar: logaritmik ölçekleme, iki veri noktası, farklı eğitim verisi miktarları.)
16. **Tokenizer, Türkçe başarının temeli.** Aynı cümle Kumru'da 8, Qwen'de 17 token. Kumru kök ve ekleri dilbilgisine uygun ayırıyor (*gurur + landır + ıyor*); Qwen "İmparatorluğu"nu beş parçaya bölüyor. İki Qwen boyutu aynı tokenizer'ı kullandığı için aralarındaki fark tamamen modelden geliyor.
17. **Benim tokenizer'ım alana bağımlı.** Vikipedi'den öğrendiği tarih cümlesini Qwen'den çok daha verimli böldü (10'a karşı 18 token), ama okul cümlesinde geride kaldı (24'e karşı 19). *Öğretmenleri* ve *toplantıya* gibi kelimeler eğitim verisinde neredeyse hiç yoktu.
18. **Ezber değil, genelleme.** Büyük modellerin, hiç görmedikleri okul metnindeki skorları Vikipedi'dekine çok yakın kaldı (−0,04 ile +0,08). Bizim model ise %27 kötüleşti (+0,300): sorun ezber değil, yalnızca Vikipedi görmüş olmasının getirdiği **dar alan**. Türkçe avantajı Vikipedi dışında da sürüyor (Kumru ile Qwen-1.7B farkı 0,140). Tek bir metinle ölçüldüğü için küçük farklar gürültü olabilir.
19. **Base model, dil ve gerçeklik.** Bir çalıştırmada Qwen3-1.7B-Base Türkçe başlayıp İngilizce bir alıştırmaya ("Which language is this? Step 1…") geçti ve gerçek bir partiye uydurma bir oy oranı atadı; başka bir çalıştırmada Türkçede kaldı ama Ankara'nın nüfusunu yanlış verip var olmayan bir "Milli Üniversitesi" uydurdu. Qwen modelleri akıcı ama güvenilmez; çok dilli model belirsiz noktalarda baskın diline kayabiliyor. Kumru ise iki çalıştırmada da akıcı ve doğru kaldı, ama turizm yazısı üslubunda; bu da eğitim verisinin büyük kısmının web metinlerinden geldiğini gösteriyor. Üretimler rastgele örneklemeyle yapıldığı için her çalıştırmada farklıdır.

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
- [x] Hazır açık modellerle (Qwen3, Kumru) aynı ölçüde karşılaştırma
- [ ] Kumru-2B'nin LoRA ile okul ve mevzuat metinlerine ince ayarı
- [ ] Daha uzun eğitim ve daha fazla veri (BPE modeli 5.000 adımda hâlâ iyileşiyordu)
- [ ] Daha büyük model ve yerel GPU'da eğitim
- [ ] MEB mevzuatı için belgeye dayalı (RAG) asistan

## Teknolojiler

Python · PyTorch · Hugging Face Transformers & Datasets · Matplotlib · Google Colab

## Kullanılan açık modeller

- [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) ve [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) — Qwen ekibi, Apache-2.0
- [Kumru-2B-Base](https://huggingface.co/vngrs-ai/Kumru-2B-Base) — VNGRS, Apache-2.0
