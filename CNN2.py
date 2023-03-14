
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# MNISTデータセットの読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 画像の前処理
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNNのモデルを構築
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# モデルのコンパイル
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# モデルの訓練
model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

# モデルの評価
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
