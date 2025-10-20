import cv2
import mediapipe as mp
import numpy as np
import uuid
import json
import os

def extract_landmarks(results):
    keypoints = []

    if results.left_hand_landmarks:
        keypoints.extend([lm.x for lm in results.left_hand_landmarks.landmark])
        keypoints.extend([lm.y for lm in results.left_hand_landmarks.landmark])
    else:
        keypoints.extend([0.0] * 42)

    if results.right_hand_landmarks:
        keypoints.extend([lm.x for lm in results.right_hand_landmarks.landmark])
        keypoints.extend([lm.y for lm in results.right_hand_landmarks.landmark])
    else:
        keypoints.extend([0.0] * 42)

    face_indices = [1, 33, 263, 61, 291, 199, 234, 454, 10, 152, 168, 8]
    if results.face_landmarks:
        for idx in face_indices:
            try:
                keypoints.append(results.face_landmarks.landmark[idx].x)
                keypoints.append(results.face_landmarks.landmark[idx].y)
            except:
                keypoints.extend([0.0, 0.0])
    else:
        keypoints.extend([0.0] * 24)

    if len(keypoints) != 108:
        return None

    return np.array(keypoints, dtype=np.float32)

def extract_keypoints_strict_fps(video_path, output_dir="results", fps_extract=30):
    os.makedirs(output_dir, exist_ok=True)
    video_id = str(uuid.uuid4())[:8]

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False)

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps_video / fps_extract))

    frames = []
    frame_count = 0

    while cap.isOpened() and len(frames) < fps_extract:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            keypoints = extract_landmarks(results)
            if keypoints is not None:
                mean = np.mean(keypoints)
                std = np.std(keypoints) + 1e-6
                normalized = (keypoints - mean) / std
                frames.append(normalized.tolist())

        frame_count += 1

    cap.release()

    while len(frames) < fps_extract:
        frames.append([0.0] * 108)

    keypoints_json = {
        "video_id": video_id,
        "frames": frames
    }

    out_path = os.path.join(output_dir, f"{video_id}_keypoints.json")
    with open(out_path, "w") as f:
        json.dump(keypoints_json, f)

    print(f"Saved: {out_path}")
    return out_path
