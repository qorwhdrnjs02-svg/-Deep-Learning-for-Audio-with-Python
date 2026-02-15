import json
from xml.parsers.expat import model
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_PATH = "data.json"

def load_data(data_path):



    with open(data_path, "r") as fp:
        #json.load() 함수를 이용해서 json 파일을 읽어들임
        #json.load() 함수는 json 파일을 읽어서 파이썬의 dict 형태로 변환해줌
        data = json.load(fp)
    #json 파일에서 "mfcc"라는 key에 대응하는 value값을 numpy array로 변환해서 inputs에 저장
    x = np.array(data["mfcc"]) 
    y = np.array(data["labels"])
    return x, y

def prepare_datasets(test_size, validation_size):
    #load data
    X, Y = load_data(DATA_PATH)

    #create train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    #create train/validation split
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size)

    #3d array -> (130, 13, 1)
    #CNN 모델은 3D 형태의 입력을 필요로 하기 때문에, 입력 데이터에 새로운 차원을 추가하여 4D 형태로 변환
    # 4차원 이유: (샘플 수) x (시간:130) x (mfcc:13) x (채널:1) 구조라서\
    # np.newaxis: 차원 추가하는 NumPy 명령어.
    X_train = X_train[..., np.newaxis] #4d array -> (new_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

def build_model(input_shape):

    #create model
    model = tf.keras.Sequential()

    #1st conv layer
    #conv2D: 2D convolutional layer, 32개의 필터,
    #(여기서 32개의 필터라고하면 32개의 서로 다른 커널이 입력 데이터에 적용된다는 의미임)
    #커널 크기는 (3,3), 
    #활성화 함수는 ReLU, 입력 형태는 input_shape
    #maxPool2D: 2D max pooling layer, 풀링 크기는 (3,3), 
    #스트라이드 크기는 (2,2), 패딩은 "same"->same은 영역 외 나머지 부분을 0으로 채우는 방식

    # 32개의 필터는 초기에는 랜덤하게 생성되지만, 학습을 거듭하면서
    # 우리가 Target으로 하는 데이터(장르 등)의 유용한 특징을 추출하는 방향으로 점점 보정됨.
    
    # 이는 마치 MLP 모델에서 가중치(Weight)를 학습하는 것과 유사하게,
    # Loss에 대한 경사하강법(Gradient Descent) 과정을 통해 필터 내부의 값들이 최적화됨.
    
    # 차이점:
    # 1. MLP 모델: 각 뉴런이 입력 데이터 전체와 연결(Fully Connected)되어,
    #    입력 크기가 커질수록 가중치 개수가 기하급수적으로 늘어남.
    # 2. CNN 모델: 각 필터가 입력 데이터의 일부분(Local Receptive Field)에 대해서만 연산을 수행하며,
    #    동일한 필터를 전체 이미지에 반복 사용(Parameter Sharing)함.
    #    -> 적은 수의 가중치로도 효율적인 학습이 가능하며, 
    #       다음 Layer의 출력 크기(Feature Map size)와 무관하게 파라미터 개수를 고정할 수 있음. 
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    # Batch Normalization (배치 정규화)
    # -------------------------------------------------------------------------
# 1. convolution과 pooling을 반복하면서 입력데이터에 대해 target에 더 가까이 다가갈 수 있도록 모델이 학습함
# 2. 그런데 그 과정에서 가중치(CNN에서는 kernel grid)를 업데이트하면서
# 3. 이전에 layer1을 거치면서 layer2에 전달됐던 데이터가 비약적으로 달라짐
# 4.그러면 연산이 복잡해지고 학습이 더뎌지며, 정확도가 떨어지니까
# 5. 이전 layer를 통과한 데이터를 batch정규화를 통해 매번 일률적인 정도의 데이터를 담을 수 있도록 함
# 6 그럼 다음 layer가 받는 데이터가 매번 일정 하니까 학습이 용이해짐
# (kernel 을 통과한 output에 담긴 수의 크기가 일정, 수의 배치가 일정하다는 뜻은 아님)

# 결론적으로 
# layer1 -> 정규화 -> layer2 -> 정규화  -> layer3 -> 정규화 -> output으로 학습한 뒤 
# 손실 함수를 거쳐서 각 conv2D kernel을 업데이트 하는 식
    model.add(tf.keras.layers.BatchNormalization())

    #2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    
    #3rd conv layer
    # Conv2D: 32개의 필터 사용, 커널 크기는 (2, 2)로 설정하여
    #         이전 층보다 더 국소적이고 세밀한 특징(Micro-features)을 추출함.
    # MaxPool2D: (2, 2) 크기로 풀링하여 데이터의 크기(Dimension)를 절반으로 줄임.
    #            (중요한 특징은 유지하면서 연산량을 줄이는 역할)
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    #flatten output and feed into dense layer
    #최종적으로 데이터를 1차원으로 펼쳐서 완전 연결층(Dense layer)에 입력
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))

    #output layer
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    return model

if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_datasets(0.25, 0.2)

    #build the CNN network
    #X_train은 4d array -> (new_samples, 130, 13, 1)
    #input_shape는 (130, 13, 1) 형태가 될 것임
    #첫번쨰 원소는 샘플의 총 개수(Batch Size의 상위 개념)일 뿐이고
    #두번째 원소는 시계열 데이터의 시간 축 길이(Time Steps / Rows)이고
    #세번째 원소는 각 시점에서의 특징 벡터 크기(Features / Columns)이고
    #네번째 원소는 채널 수(Channels)이니까
    input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape=input_shape)
    #compile the network

    #train the CNN network

    #evaluate the CNN on the test set

    #make predictions on a sample