# Med-Guard

Med-Guard, bir sağlık/tıp asistanına gelen kullanıcı isteklerini **güvenli (safe) / güvensiz (unsafe)** olarak sınıflandıran ve isteğin **niyetini (intent)** dokuz alt kategoriye ayıran bir güvenlik (guardrail) modelidir. Amaç; kendine zarar verme riski, zararlı tedavi önerisi talebi, manipülasyon gibi riskli girdileri, iyi huylu genel sorulardan ve kronik/akut tıbbi durumlardan ayırt edebilmektir.

## Yöntem

Model, çok dilli cümle gömme (sentence embedding) vektörleri üzerine kurulu bir **işaretli graf (signed graph) tabanlı embedding propagation** yaklaşımı kullanır:

1. **Embedding** — İstekler `paraphrase-multilingual-mpnet-base-v2` ile vektörleştirilir (`sentence-transformers`).
2. **İşaretli graf inşası** — Eğitim setindeki örnekler arasında kosinüs benzerliğine göre k-NN grafı kurulur:
   - **Pozitif kenarlar**: aynı etikete sahip ve yüksek benzerlikteki örnekler arasında (pull).
   - **Negatif kenarlar**: önceden tanımlı "karıştırılabilir" niyet çiftleri (`confusion_pairs`) arasında, zıt etiketli ve yüksek benzerlikteki zor negatifler arasında (push).
3. **Sinir ağı** — Paylaşılan bir gövdeden (`Linear → ReLU → Dropout`) iki başa dallanan bir ağ: güvenlik (binary) başı ve niyet (9 sınıf) başı.
4. **Embedding propagation** — Öğrenilen özellikler, pozitif graf üzerinden yayılır (pull) ve negatif graf ile bu yayılım geri bastırılır (push); böylece benzer örnekler etiket uzayında birbirine yaklaşır, çelişen örnekler birbirinden uzaklaşır.
5. **Kayıp fonksiyonu** — Ağırlıklandırılmış BCE (güvenlik), sınıf ağırlıklı CE (niyet), pull-loss, margin tabanlı push-loss ve propagation-consistency loss'un toplamı.
6. **Inductive çıkarım** — Test/kalibrasyon setindeki görülmemiş örnekler için, eğitim setindeki en yakın komşulara dayalı bir mahalle inşa edilip aynı propagation mantığı uygulanır (transduktif değil, inductive).

## Niyet (intent) kategorileri

| Sınıf | Açıklama |
|---|---|
| `benign_general` | Genel, zararsız istek |
| `medication_related` | İlaçla ilgili istek |
| `mild_emotional_stress` | Hafif duygusal sıkıntı |
| `chronic_condition` | Kronik durumla ilgili istek |
| `non_acute_injury` | Akut olmayan yaralanma |
| `acute_medical_risk` | Akut tıbbi risk |
| `harmful_treatment` | Zararlı tedavi talebi |
| `manipulation` | Manipülatif istek |
| `self_harm_risk` | Kendine zarar verme riski |

## Proje yapısı

```
Med-Guard/
├── data/
│   └── dataset_shuffled.csv  # Veri seti
├── notebooks/
│   └── guard.ipynb       # İnce orkestrasyon: src/ modüllerini çağırır, sonuçları analiz eder
├── src/
│   ├── config.py         # Hiperparametreler, etiket/niyet haritaları
│   ├── data.py            # Veri yükleme, embedding çıkarımı, train/cal/test ayrımı
│   ├── graph.py           # İşaretli graf (signed graph) inşası
│   ├── propagation.py     # Transduktif ve inductive embedding propagation
│   ├── model.py            # SignedGraphSafetyModel mimarisi
│   ├── losses.py           # Pull/push ve propagation-consistency kayıpları
│   ├── train.py             # Eğitim döngüsü
│   └── evaluate.py          # Çıkarım (inference)
├── requirements.txt
├── LICENSE
└── README.md
```

Ağır/tekrar kullanılabilir mantık `src/` altında modüller halinde tutulur; `notebooks/guard.ipynb` bu modülleri import edip çalıştıran ve sonuçları (metrikler, propagation etkisi, yanlış negatif analizi) satır satır gösteren ince bir katmandır.

## Kurulum

```bash
git clone https://github.com/caglafikir/Med-Guard.git
cd Med-Guard
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS / Linux
pip install -r requirements.txt
```

Veri seti `data/dataset_shuffled.csv` altında yer alır ve en az şu sütunları içerir:

| Sütun | Açıklama |
|---|---|
| `request` | Kullanıcı isteğinin metni |
| `safety_level` | `safe` / `unsafe` |
| `intent` | Yukarıdaki 9 niyet kategorisinden biri |

## Kullanım

```bash
jupyter notebook notebooks/guard.ipynb
```

Notebook; `src/` modüllerini kullanarak veriyi yükler, embedding'leri çıkarır, işaretli grafı kurar, modeli eğitir ve test setinde `classification_report`, karışıklık matrisi ile yanlış negatif (false negative) analizleri üretir.

`src/` modülleri bağımsız olarak da kullanılabilir, örneğin:

```python
from src.config import Config
from src import data

cfg = Config()
df = data.load_dataset(cfg.DATA_PATH)
```

## Konfigürasyon

Önemli hiperparametreler `src/config.py` içindeki `Config` dataclass'ında tanımlıdır: graf komşuluk sayısı (`K_GRAPH`), pozitif/negatif benzerlik eşikleri, kayıp ağırlıkları (`LAMBDA_*`), propagation katsayıları (`PROP_ALPHA_*`) ve eğitim hiperparametreleri (`LR`, `EPOCHS`, vb.).

## Lisans

Bu proje [MIT lisansı](LICENSE) ile lisanslanmıştır.

## Sorumluluk reddi

Bu proje bir araştırma/prototip çalışmasıdır ve gerçek tıbbi karar destek sistemlerinde doğrudan kullanılmak üzere tasarlanmamıştır.
