

# 🤲 Arabic Sign Deep Learning: Hybrid Spatio-Temporal Recognition Model

**Shatha3344/arabic\_sign\_deeplearning**

مشروع بحثي وتطبيقي يهدف إلى التعرف على لغة الإشارة العربية (ArSL) باستخدام نموذج هجين متقدم يعتمد على شبكات المُحوّلات (Transformers) والمعالم الحركية المستخلصة.


## 🚀 نظرة عامة على المشروع (Project Overview)

تم تطوير نظام التعرف على الإشارة بناءً على بنية **مكانية-زمانية هجينة (Hybrid Spatio-Temporal Architecture)**. يعالج هذا النموذج التسلسلات الزمنية لنقاط المفتاح (Keypoints) المستخلصة عبر MediaPipe، مما يضمن دقة عالية في تصنيف الإشارات الديناميكية.


## 3\. البنية الهيكلية للنموذج (Model Architectural Structure)

  * **الشكل (`architecture.png`):** يوضح البنية الهندسية للنموذج الهجين `hybird_v3_toptransfprm`.
      * **المدخلات:** تسلسلات زمنية من المعالم الحركية (Keypoint Coordinates) المُعيرة (Normalized) والمُستخلصة من الإشارة.
      * **المُحوّل الزماني (Temporal Transformer):** المكون الرئيسي الذي يستخدم **آليات الانتباه الذاتي (Self-Attention)** لالتقاط الاعتماديات طويلة المدى في حركة الإشارة عبر الزمن.

-----

## 4\. منهجية التدريب والتحسين (Training Methodology)

  * **الشكل (`asset/image/alogthims.jpg`):** يعرض مخطط سير عمل **(Workflow Diagram)** لعملية التدريب والتقييم.
      * تشمل المنهجية: استخلاص المعالم (`asset/image/pipe.png`) $\rightarrow$ تسوية البيانات (`scaler.joblib`) $\rightarrow$ تدريب النموذج الهجين $\rightarrow$ التقييم المستمر (`asset/image/train.png`).
  * **ملحقات التدريب:**
      * **النموذج المُدرّب:** `arabic_sign_deeplearning/models/hybird_v3_toptransfprm/hybrid_model_final_108.keras`
      * **ترميز الفئات:** `arabic_sign_deeplearning/models/hybird_v3_toptransfprm/label_encoder.joblib`

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

*(ملاحظة: قد تحتاج إلى حزم إضافية بناءً على ملف `arabic-words-sign-language-detection (2).ipynb`)*

### 3\. تحميل النموذج والملحقات

تأكد من وجود الملفات التالية داخل المسار الصحيح:

  * `arabic_sign_deeplearning/models/hybird_v3_toptransfprm/hybrid_model_final_108.keras`
  * `arabic_sign_deeplearning/models/hybird_v3_toptransfprm/label_encoder.joblib`
  * `arabic_sign_deeplearning/models/hybird_v3_toptransfprm/scaler.joblib`

### 4\. تشغيل التطبيق (Run the Application)

#### أ. تشغيل الواجهة الأمامية (Web Interface)

إذا كان التطبيق يعتمد على نموذج `JavaScript/TensorFlow.js` يعمل في المتصفح:

1.  انتقل إلى المجلد: `cd arabic_sign_deeplearning/web`
2.  افتح ملف `index.html` مباشرة في متصفح يدعم WebCam (Chrome/Firefox).

#### ب. تشغيل الكود المحلي (Local Execution)

إذا كنت تستخدم Python (مثل Flask/Django) لتقديم النموذج:

1.  **لتشغيل التدريب/التحليل:** افتح الملف `notebooks/arabic-words-sign-language-detection (2).ipynb` في Jupyter Notebook أو VS Code.
2.  **لتشغيل الخدمة (Service):** قم بتنفيذ ملف Python الرئيسي الذي يقوم بتحميل النموذج وربطه بالكاميرا (يفترض وجود ملف مثل `app.py`):
    ```bash
    python arabic_sign_deeplearning/main_app.py 
    ```
    (استبدل `main_app.py` بالاسم الفعلي لملف تشغيل الخدمة لديك).

-----

## 📊 النتائج والتحليل (Results and Analysis)

  * **الدقة (`acc.png`)** و **منحنيات التدريب (`train.png`)** تؤكد فعالية المنهجية.
  * **منصة التطبيق (`screen_platform.jpg`)** تُظهر واجهة المستخدم التفاعلية.
