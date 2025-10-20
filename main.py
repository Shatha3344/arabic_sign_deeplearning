import os
import joblib
import uuid
import json
import numpy as np
import mediapipe as mp
import cv2
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from keras.layers import Masking, MultiHeadAttention, LayerNormalization
from keras.utils import get_custom_objects
from werkzeug.utils import secure_filename
from utils.landmark_extractor import extract_landmarks

# إعداد Flask
app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER='uploads',
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,
    ALLOWED_EXTENSIONS={'mp4', 'avi', 'mov'}
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# تسجيل الطبقات المخصصة
get_custom_objects().update({
    "Masking": Masking,
    "MultiHeadAttention": MultiHeadAttention,
    "LayerNormalization": LayerNormalization,
})

# تحميل النموذج والمكونات
MODEL_PATH = "C:/Users/shath/Documents/sign_arabic/model_assets/hybird_v3_toptransfprm/hybrid_model_final_108.keras"
SCALER_PATH = "C:/Users/shath/Documents/sign_arabic/model_assets/hybird_v3_toptransfprm/scaler.joblib"
ENCODER_PATH = "C:/Users/shath/Documents/sign_arabic/model_assets/hybird_v3_toptransfprm/label_encoder.joblib"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# إعداد MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

NUM_FRAMES = 30
NUM_FEATURES = 108

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = holistic.process(frame)
        keypoints = extract_landmarks(results)

        if keypoints is not None and keypoints.shape[0] == NUM_FEATURES:
            frames.append(keypoints)
    
    cap.release()

    if not frames:
        return None

    while len(frames) < NUM_FRAMES:
        insert_index = len(frames) // 2
        frames.insert(insert_index, np.zeros(NUM_FEATURES, dtype=np.float32))

    if len(frames) > NUM_FRAMES:
        start_index = (len(frames) - NUM_FRAMES) // 2
        frames = frames[start_index:start_index + NUM_FRAMES]

    return np.stack(frames).reshape(1, NUM_FRAMES, NUM_FEATURES)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/translate')
def translate():
    return render_template('translate.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/predict', methods=['POST'])
def predict_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'لم يتم تقديم ملف فيديو'}), 400

        video_file = request.files['video']
        if video_file.filename == '' or not allowed_file(video_file.filename):
            return jsonify({'error': 'نوع الملف غير مسموح أو الاسم فارغ'}), 400

        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        raw_features = process_video(video_path)
        os.remove(video_path)

        if raw_features is None:
            return jsonify({'error': 'فشل في استخراج النقاط من الفيديو'}), 500

        # إنشاء ID عشوائي
        video_id = str(uuid.uuid4())[:8]
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # حفظ ملف keypoints الخام قبل التطبيع
        raw_json_path = os.path.join(results_dir, f"{video_id}_keypoints.json")
        with open(raw_json_path, "w") as f:
            json.dump({
                "video_id": video_id,
                "frames": raw_features.reshape(NUM_FRAMES, NUM_FEATURES).tolist()
            }, f)

        # تطبيع النقاط باستخدام الـ Scaler المُدرّب
        flat = raw_features.reshape(-1, NUM_FEATURES)
        non_zero = np.any(flat != 0, axis=1)
        if np.any(non_zero):
            flat[non_zero] = scaler.transform(flat[non_zero])
        features_scaled = flat.reshape(1, NUM_FRAMES, NUM_FEATURES)

        pred = model.predict(features_scaled)
        pred_class = np.argmax(pred)
        label = label_encoder.inverse_transform([pred_class])[0]
        confidence = float(np.max(pred))

        # حفظ نتيجة التنبؤ
        result_path = os.path.join(results_dir, f"{video_id}_label.json")
        with open(result_path, "w") as f:
            json.dump({
                "video_id": video_id,
                "prediction": label,
                "confidence": confidence
            }, f)

        return jsonify({
            "video_id": video_id,
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({'error': f'خطأ غير متوقع: {str(e)}'}), 500

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'طلب غير صالح'}), 400

@app.errorhandler(413)
def request_too_large(e):
    return jsonify({'error': 'الفيديو أكبر من الحجم المسموح (50MB)'}), 413

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'error': 'حدث خطأ داخلي أثناء الترجمة'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
