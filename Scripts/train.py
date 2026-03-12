import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split

TIMESTEPS = 30 # 1秒間にどれだけのデータを処理するか
FEATURES = 33 # 特徴量の数(ランドマークx3)

try:
    X_focus = np.load("./data/focus_data.npy") # 集中
    X_unfocus = np.load("./data/unfocus_data.npy") # 非集中
    X = np.concatenate([X_focus, X_unfocus], axis=0)
    y = np.concatenate([np.ones(len(X_focus)), np.zeros(len(X_unfocus))])
except Exception as e:
    print(f"ファイルが無いか破損しています {e}")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

print(f"学習用データの形状: {X_train.shape}")
print(f"テスト用データの形状: {X_test.shape}")

model = Sequential()

#Conv層
model.add(Conv1D(
    input_shape=(TIMESTEPS, FEATURES),
    filters=16,
    kernel_size=3,
    activation="relu"
))

model.add(Conv1D(
    filters=32,
    kernel_size=3,
    activation="relu"
))

# Pooling層
model.add(MaxPooling1D(
    pool_size=2
))

# LSTM層
model.add(LSTM(
    units=32,
    return_sequences=True
))

model.add(LSTM(
    units=16,
    # 最終的に１つの出力にする
    return_sequences=False
))

# Dense(全結合層)
model.add(Dense(
    units=32,
    activation="relu"
))

# Dropout層
model.add(Dropout(.5))

# 出力層: 0(非集中)~1(集中)
model.add(Dense(1, activation="sigmoid"))

# モデルのコンパイル
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print("学習開始します")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

model.save("./models/model.keras")
print("モデル保存完了")