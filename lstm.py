import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from cnn_genre_classifier import predict

DATA_PATH = "data.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y

def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def prepare_datasets(test_size, validation_size):
    X, Y = load_data(DATA_PATH)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size)

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

    #RNN 모델은 3D 형태의 입력을 필요로 하기 때문에
    #CNN 모델과 달리 차원을 추가할 필요가 없음.

def build_model(input_shape):
    #Generate RNN-LSTM model
    #:param input_shape: (tuple) : shape of input data
    #:return model: RNN-LSTM model
  
    model = tf.keras.Sequential()

    #2LSTM 레이어
    #return_sequences 는 sequence-to-sequence / sequence-to-vector 모델을 결정하는 매개변수
    #ture로 섧정하면 sequence-to-sequence 모델이 되어 각 타임스텝마다 출력이 생성되고
    #False로 설정하면 sequence-to-vector 모델이 되어 마지막 타임스텝의 출력만 생성됨.
    #(17강 참조)
    #sequence-to-sequence 모델은 시퀀스의 각 타임스텝에 대해 출력이 필요한 작업
    #(음악에 경우 시간에 따른 음파의 변화를 관측하는 것이 중요하므로 sequence-to-sequence 모델이 적합)
    #sequence-to-vector 모델은 시퀀스 전체에 대한 단일 출력이 필요한
    #(음악 역시 전체 데이터를 훑은 후 최종적으로 장르를 분류하는 것을 목표로 한다면
    #sequence-to-vector 모델을 사용할 수 있지만 정확도가 떨어지고 학습이 더 어려울 수 있음)
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))

    #Dense 레이어
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    #과적합 방지 위해 Dropout 레이어 추가
    model.add(tf.keras.layers.Dropout(0.3))

    #출력 레이어
    #10개의 장르로 분류하기 위해 10개의 뉴런을 가진 출력
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    return model

def predict(model, X, Y):
    #model이 예측한 확률 분포에서 가장 높은 확률을 가진 인덱스를 예측된 레이블로 간주
    X = X[np.newaxis, ...] #4D array -> (1, 130, 13, 1)
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)

    print(f"Expected index: {Y}, Predicted index: {predicted_index}")

if __name__ == "__main__":
    # get the train, validation, test sets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_datasets(0.25, 0.2)

    # build the CNN network
    #3차원 input을 가졌던 CNN 모델과 달리
    #RNN 모델이므로 마찬가지로 2차원 입력 형태를 가지는 input_shape을 정의
    input_shape = (X_train.shape[1], X_train.shape[2]) #130, 13
    model = build_model(input_shape)

    # compile the network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # train the CNN
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=32, epochs=30)

    plot_history(history)

    #evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=2)
    print(f"Test error: {test_error}, Test accuracy: {test_accuracy}")

    #make predictions on a sample
    # 4. 개별 샘플 테스트
    # -------------------------------------------------------------------------
    # 전체 테스트셋(X_test) 중에서 100번째 데이터를 하나 콕 집어서 꺼냄.
    # X_test[100]의 형태는 (130, 13, 1)임. (Batch 차원이 빠진 상태)
    X = X_test[100]
    Y = Y_test[100]
    predict(model, X, Y)