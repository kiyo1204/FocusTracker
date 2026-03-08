# coding: utf8

import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import time
import sys

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model


# 初期設定
cap = cv2.VideoCapture(0)
TIMESTEPS = 30 

# 保存先フォルダとモデルの準備
os.makedirs("./data", exist_ok=True)
os.makedirs("./Pose_model", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# 最新のTasks APIで使うAIモデル（pose_landmarker_lite.task）を自動ダウンロード
model_path = './Pose_model/pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("最新のMediaPipe用AIモデルをダウンロードしています...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    print("ダウンロード完了！")

# Tasks API の設定
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE, # フレームごとに処理するモード
        num_poses=1                            # 検出する最大人数
    )
detector = vision.PoseLandmarker.create_from_options(options)


def pred():
    try:
        model = load_model("./models/model.keras")
    except Exception as e:
        print("モデルを作成してください")
        detector.close()
        exit()

    print("予測を開始します... ('@'キーで終了)")
    time.sleep(3)

    sequence_data = []
    current_status = "None"
    status_color = (255, 255, 255)

    # データ収集ループ
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe専用の画像フォーマットに変換
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 推論の実行
        results = detector.detect(mp_image)

        # 姿勢(顔の輪郭)が検出された場合（results.pose_landmarks に結果が入る）
        if results.pose_landmarks:
            # 1人目の検出結果を取得
            landmarks = results.pose_landmarks[0]
            
            target_indices = range(11)
            frame_features = []

            for index in target_indices:
                lm = landmarks[index]
                px, py = int(lm.x * width), int(lm.y * height)

                cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)
                frame_features.extend([lm.x, lm.y, lm.z])

            sequence_data.append(frame_features)

            if len(sequence_data) == TIMESTEPS:
                # 入力の形に変換(1, 30, 33)
                input_data = np.expand_dims(sequence_data, axis=0)

                #予測
                prediction = model.predict(input_data, verbose=0)[0][0]

                if prediction <= .5:
                    current_status = f"UnFocus({prediction:.2f})"
                    status_color = (0, 0, 255)
                else:
                    current_status = f"Focus({prediction:.2f})"
                    status_color = (0, 255, 0)

                sequence_data.pop(0)

        cv2.putText(frame, current_status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        cv2.imshow("Real-Time Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("@"): 
            print("プログラムを停止します")
            detector.close()
            break

    # 終了処理とデータ保存
    cap.release()
    cv2.destroyAllWindows()

def create_features(file_name):
    sequence_data = []
    all_samples = []

    # データ収集ループ
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe専用の画像フォーマットに変換
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 推論の実行
        results = detector.detect(mp_image)

        # 姿勢(顔の輪郭)が検出された場合（results.pose_landmarks に結果が入る）
        if results.pose_landmarks:
            # 1人目の検出結果を取得
            landmarks = results.pose_landmarks[0]
            
            target_indices = range(11)
            frame_features = []

            for index in target_indices:
                lm = landmarks[index]
                px, py = int(lm.x * width), int(lm.y * height)

                cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)
                frame_features.extend([lm.x, lm.y, lm.z])

            sequence_data.append(frame_features)

            if len(sequence_data) == TIMESTEPS:
                all_samples.append(sequence_data)
                sequence_data = []
                print(f"データ保存完了: 現在のサンプル数 {len(all_samples)}\n")

        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("@"):
            # 終了処理とデータ保存
            cap.release()
            cv2.destroyAllWindows()

            print("- プログラムを停止します\n- 取得データを保存しますか?\n")
            if input("- 保存するなら'YES'を押してください\n") == "YES":  
                print("** データを保存します **")
                if len(all_samples) > 0:
                    X_data = np.array(all_samples)
                    print(f"\n データの最終形状: {X_data.shape}")
                    
                    np.save(file_name, X_data)
                    print(f"✅ {file_name} に保存しました！")
                else:
                    print("\n⚠️ サンプルが1つも取得できませんでした。保存をスキップします。")
                    detector.close()
                    break
            else:
                print("** 保存せずに終了します **")
            detector.close()
            break

if __name__ == "__main__":
    print("=== 集中度データ収集ツール ===")
    if len(sys.argv) > 1:
        number = sys.argv[1]
    else:
        print("error: 引数が指定されていません")
        detector.close()
        exit()

    if number == "1":
        file_name = "./data/focus_data.npy"
        print("【集中】のデータを収集します。")
        time.sleep(3)
        print("記録を開始します... ('@'キーで終了)")
        create_features(file_name)
    elif number == "2":
        file_name = "./data/unfocus_data.npy"
        print("【非集中】のデータを収集します。")
        time.sleep(3)
        print("記録を開始します... ('@'キーで終了)")
        create_features(file_name)
    elif number == "3":
        pred()
    else:
        print("1か2か3を入力してください。\nプログラムを終了します。")
        detector.close()
        exit()
