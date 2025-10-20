
-----

# 📖 Arabic Sign Deep Learning: Hybrid Spatio-Temporal Recognition Model

## نظام التعرف على لغة الإشارة العربية: نموذج مكاني-زماني هجين

**المؤلف:** Shatha3344
**المستودع:** `Shatha3344/arabic_sign_deeplearning`

مشروع بحثي وتطبيقي يهدف إلى التعرف على لغة الإشارة العربية (ArSL) باستخدام نموذج هجين متقدم يعتمد على شبكات المُحوّلات (Transformers) والمعالم الحركية المستخلصة.

-----

## 🚀 نظرة عامة على المشروع (Project Overview)

تم تطوير نظام التعرف على الإشارة بناءً على بنية **مكانية-زمانية هجينة (Hybrid Spatio-Temporal Architecture)**. يعالج هذا النموذج التسلسلات الزمنية لنقاط المفتاح (Keypoints) المستخلصة عبر MediaPipe، مما يضمن دقة عالية في تصنيف الإشارات الديناميكية.

-----

## 3\. البنية الهيكلية للنموذج (Model Architectural Structure)

  * **الشكل 1: البنية الهندسية للنموذج الهجين hybird_v3_toptransfprm`**``)
    \<img src="arabic_sign_deeplearning/asset/image/architecture.png" alt="Detailed architecture of the hybrid Spatio-Temporal Transformer model" style="max-width: 90%; display: block; margin: 0 auto;"\>
      * **المدخلات:** تسلسلات زمنية من المعالم الحركية (Keypoint Coordinates) المُعيرة (Normalized) والمُستخلصة من الإشارة.
      * **المُحوّل الزماني (Temporal Transformer):** المكون الرئيسي الذي يستخدم **آليات الانتباه الذاتي (Self-Attention)** لالتقاط الاعتماديات طويلة المدى في حركة الإشارة عبر الزمن.

-----

## 4\. منهجية التدريب والتحسين (Training Methodology)

  * **الشكل 2: مخطط سير عمل التدريب الشامل.** (`alogthims.jpg`)
    \<img src="alogthims.jpg" alt="Training methodology and workflow diagram showing data pipeline" style="max-width: 90%; display: block; margin: 0 auto;"\>

      * تشمل المنهجية: استخلاص المعالم (`pipe.png`) $\rightarrow$ تسوية البيانات (`scaler.joblib`) $\rightarrow$ تدريب النموذج الهجين $\rightarrow$ التقييم المستمر (`train.png`).

  * **مرحلة استخلاص المعالم (Keypoint Extraction):**

      * توضح الصورة التالية المسار الذي يتم فيه تحويل الصورة المرئية إلى بيانات هيكلية باستخدام MediaPipe. (`pipe.png`)

    \<img src="pipe.png" alt="MediaPipe keypoint extraction pipeline for Arabic Sign Language" style="max-width: 90%; display: block; margin: 0 auto;"\>

  * **ملحقات التدريب:**

      * **النموذج المُدرّب:** `arabic_sign_deeplearning/models/hybird_v3_toptransfprm/hybrid_model_final_108.keras`
      * **ترميز الفئات:** `arabic_sign_deeplearning/models/hybird_v3_toptransfprm/label_encoder.joblib`

-----

## 5\. النتائج والتحليل (Results and Analysis)

يتم عرض النتائج الرئيسية التي تثبت كفاءة وفعالية النموذج:

  * **منحنيات التدريب (Training Curves):** (`train.png`)

      * تُظهر المنحنيات سلوك تقارب مستقر بين دقة التدريب ودقة التحقق.

    \<img src="train.png" alt="Training loss and accuracy curves over epochs" style="max-width: 90%; display: block; margin: 0 auto;"\>

  * **مقاييس الأداء (Performance Metrics):** (`acc.png`)

      * ملخص لأفضل دقة تم الوصول إليها على مجموعة الاختبار (Test Set).

    \<img src="acc.png" alt="Classification accuracy metrics and confusion matrix" style="max-width: 90%; display: block; margin: 0 auto;"\>

  * **نتائج التحقق العملي (Practical Validation):** (`result.png`)

      * لقطة تعرض نتائج تنبؤات النموذج الناجحة على بيانات اختبار حقيقية.

    \<img src="result.png" alt="Live test results and model predictions on test data" style="max-width: 90%; display: block; margin: 0 auto;"\>

  * **منصة التنفيذ (Deployment Platform):** (`screen_platform.jpg`)

      * تُظهر واجهة المستخدم التفاعلية التي تتيح اختبار النموذج في الوقت الفعلي.

    \<img src="screen\_platform.jpg" alt="Screenshot of the web application user interface" style="max-width: 90%; display: block; margin: 0 auto;"\>

-----

## 📁 هيكلة الملفات والمجلدات (File Structure)

تتبع هيكلة المستودع التوزيع المنطقي التالي:

```
arabic_sign_deeplearning/
├── arabic_sign_deeplearning/ 
│   ├── models/
│   │   └── hybird_v3_toptransfprm/
│   │       ├── hybrid_model_final_108.keras # ملف النموذج المُدرّب
│   │       ├── label_encoder.joblib         # لفك الترميز
│   │       └── scaler.joblib                # لمعايرة البيانات
│   ├── web/                   # تطبيق الويب (الواجهة الأمامية)
│   │   ├── index.html         # الواجهة الرئيسية
│   │   ├── style.css          # ملفات CSS (منقولة من SASS)
│   │   └── style.scss         # ملفات SASS المصدر
│   └── (ملفات Python/Scripts الأخرى لتشغيل النموذج)
├── notebooks/
│   └── arabic-words-sign-language-detection (2).ipynb # كود التدريب والتحليل
├── acc.png                    # دقة النموذج
├── architecture.png           # هيكلية النموذج (شكل رقم 1)
├── alogthims.jpg              # منهجية التدريب (شكل رقم 2)
├── pipe.png                   # استخلاص المعالم
├── train.png                  # منحنيات التدريب
└── README.md
```

-----

## ▶️ كيفية التشغيل (Getting Started)

لتشغيل المشروع والتفاعل مع النموذج، اتبع الخطوات التالية:

### 1\. الاستنساخ (Clone the Repository)

```bash
git clone https://github.com/Shatha3344/arabic_sign_deeplearning.git
cd arabic_sign_deeplearning
```

### 2\. إعداد البيئة (Environment Setup)

يتطلب المشروع بيئة Python (يُفضل استخدام Anaconda/Conda) وحزم التعلم العميق:

```bash
# إنشاء بيئة جديدة (اختياري)
conda create -n arsign python=3.9
conda activate arsign

# تثبيت الحزم المطلوبة
pip install tensorflow keras scikit-learn joblib mediapipe 
```

### 3\. تحميل النموذج والملحقات

تأكد من وجود جميع الملفات الخاصة بالنموذج والمُعالج (المذكورة في قسم هيكلة الملفات) في المسارات الصحيحة.

### 4\. تشغيل التطبيق (Run the Application)

#### أ. تشغيل الواجهة الأمامية (Web Interface)

1.  انتقل إلى المجلد: `cd arabic_sign_deeplearning/web`
2.  افتح ملف **`index.html`** مباشرة في متصفح يدعم WebCam (Chrome/Firefox).

#### ب. تشغيل الكود المحلي (Local Execution)

1.  **لتشغيل التدريب/التحليل:** افتح الملف **`notebooks/arabic-words-sign-language-detection (2).ipynb`** في Jupyter Notebook أو VS Code.
2.  **لتشغيل الخدمة (Service):** قم بتنفيذ ملف Python الرئيسي (على افتراض وجود ملف تشغيل أساسي):
    ```bash
    python arabic_sign_deeplearning/main_app.py 
    ```
    (استبدل `main_app.py` بالاسم الفعلي لملف تشغيل الخدمة لديك).
