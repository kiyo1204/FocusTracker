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
cap = cv2.VideoCapture(0) # カメラ読み込み
TIMESTEPS = 30 # １秒間に取得するデータ数
ALPHA = .5 # 座標の平滑化係数
JUMP_THRESHOLD = .2 # 1フレームで動ける限界値(画面に対する割合)

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

    print("予測を開始します")
    time.sleep(3)

    sequence_data = []
    current_status = "None"
    status_color = (255, 255, 255)
    prev_landmarks = None

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

            # 鼻をインデックス0として座標取得
            nose_x = landmarks[0].x
            nose_y = landmarks[0].y
            nose_z = landmarks[0].z
            
            target_indices = [0, 2, 5, 8, 7, 9, 10, 11, 12, 15, 16]
            frame_features = []

            for index in target_indices:
                lm = landmarks[index]
                # 画面に円を出力させるための計算
                px, py = int(lm.x * width), int(lm.y * height)

                # 鼻を基準にした相対座標の計算
                rel_x = lm.x - nose_x
                rel_y = lm.y - nose_y
                rel_z = lm.z - nose_z

                cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)
                frame_features.extend([rel_x, rel_y, rel_z])

            frame_features = np.array(frame_features)

            # ノイズ対策(微妙な振動)
            if prev_landmarks is None:
                smoothed_features = frame_features
            else:
                smoothed_features = ALPHA * frame_features + (1 - ALPHA) * prev_landmarks

            prev_landmarks = smoothed_features

            sequence_data.append(smoothed_features.tolist())

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
        cv2.putText(frame, "Press '@' to exit", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Real-Time Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("@"): 
            print("--- プログラムを停止します ---")
            break

    # 終了処理とデータ保存
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

def create_features(file_name):
    sequence_data = []
    all_samples = []
    prev_landmarks = None

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

            # Visibilityが.5以下だったら不正確なデータとして破棄
            if landmarks[0].visibility < .5:
                continue

            # 鼻をインデックス0として座標取得
            nose_x = landmarks[0].x
            nose_y = landmarks[0].y
            nose_z = landmarks[0].z
            
            target_indices = [0, 2, 5, 8, 7, 9, 10, 11, 12, 15, 16]
            frame_features = []
            # 外れ値かどうか
            is_outlier = False

            for index in target_indices:
                lm = landmarks[index]
                # 画面に円を出力させるための計算
                px, py = int(lm.x * width), int(lm.y * height)

                # 鼻を基準にした相対座標の計算
                rel_x = lm.x - nose_x
                rel_y = lm.y - nose_y
                rel_z = lm.z - nose_z

                cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)
                frame_features.extend([rel_x, rel_y, rel_z])

            if is_outlier:
                continue

            frame_features = np.array(frame_features)

            # ワープ(読み取りエラー)の判定
            if prev_landmarks is not None:
                # 全座標のずれの平均
                distance = np.mean(np.abs(frame_features - prev_landmarks))

                # 平均が閾値を超えたら破棄
                if distance > JUMP_THRESHOLD:
                    print("⚠読み取り誤検知のため破棄")
                    continue

                # ノイズ対策(微妙な振動)
                smoothed_features = ALPHA * frame_features + (1 - ALPHA) * prev_landmarks
            else:
                smoothed_features = frame_features

            prev_landmarks = smoothed_features

            sequence_data.append(smoothed_features.tolist())

            if len(sequence_data) == TIMESTEPS:
                all_samples.append(sequence_data)
                sequence_data = []
                print(f"✅データ保存完了: 現在のサンプル数 {len(all_samples)}")

        cv2.putText(frame, "Press '@' to exit", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("@"):
            break

        cap.release()
        cv2.destroyAllWindows()

        print("プログラムを停止します")
        print("取得データを保存しますか? 保存するなら'YES'を押してください")
        if input() == "YES": 
            print("*** データを保存します ***")
            if len(all_samples) > 0:
                X_data = np.array(all_samples)

                if os.path.exists(file_name):
                    print("既存ファイルに結合しますか?")
                    print("保存するなら'YES'を押してください")
                    if input() == "YES":
                        print("*** 既存データに結合します ***")
                        existing_data = np.load(file_name)
                        X_data = np.concatenate((existing_data, X_data), axis=0)
                        print("✅ 既存データと結合しました")
                    
                print(f"- データの最終形状: {X_data.shape}")
                np.save(file_name, X_data)
                print("✅ 保存しました！")
            else:
                print("⚠️ サンプルが1つも取得できませんでした。保存をスキップします. ")
                
            print("--- 終了します ---")
            detector.close()
            break

if __name__ == "__main__":
    print("=== 集中度データ収集ツール ===")
    if len(sys.argv) > 1:
        number = sys.argv[1]
    else:
        detector.close()
        exit()

    if number == "1":
        file_name = "./data/focus_data.npy"
        print("【集中】のデータを収集します. ")
        time.sleep(3)
        print("記録を開始します")
        create_features(file_name)
    elif number == "2":
        file_name = "./data/unfocus_data.npy"
        print("【非集中】のデータを収集します。")
        time.sleep(3)
        print("記録を開始します")
        create_features(file_name)
    elif number == "3":
        pred()
    else:
        print("1か2か3を入力してください. \nプログラムを終了します。")
        detector.close()
        exit()
