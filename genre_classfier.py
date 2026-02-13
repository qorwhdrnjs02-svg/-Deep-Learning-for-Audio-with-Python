import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


DATASET_PATH = "data.json"

#데이터 패스를 인자로 받는 관습
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        #json 파일은 왠만하면 6개의 데이터 형태를 갖게됨
        #1. { "key": "value" } (Object)
        # -> dict (딕셔너리)	중괄호를 사용하며, 키와 값의 쌍으로 저장됨
        #2. [ 1, 2, 3 ] (Array)	
        # -> list (리스트)	대괄호를 사용하며, 순서가 있는 데이터 모음임
        #3. "string" (String) 
        # -> str (문자열)	반드시 큰따옴표(")를 사용해야 함
        #4. 123 / 12.3 (Number)	
        # -> int / float	정수와 실수를 구분하여 인식함
        #5. true / false
        # -> True / False (Bool)	첫 글자가 대문자로 변환됨 (파이썬 관례)
        #6. null	
        # -> None	데이터가 없음을 뜻하며 파이썬의 None과 대응됨
        data = json.load(fp)
        #그렇게 json.load 하면 파일을 읽어 data 라는 이름으로 파일 형태에
        #대응하는 정보의 형태를 저장하게 됨

    #convert lists into numpy arrays
    #data는 dict 형태
    #특히나 우리는 key:value의 정보에서 value 값을 list로 저장했음
    #단 json은 python 자체의 list를 이용하기 따문에
    # "mfcc", "labels"라는 key에 대응하는 value값으로서의 list를
    # np.array 즉 numpy array로 저장함 
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    
    return inputs, targets

def plot_history(history):

    flg, axs = plt.subplots(2)

    #create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    #create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")  
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    #load data
    #함수 호출해서 상수값으로 정의한 path 인자로 입력,
    #return 값을 차례로 input과 target에 대응
    lnputs, targets = load_data(DATASET_PATH)

    #split the data into tarin and test sets
    # train 모델과 test 모델을 7:3으로 나누고
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
                                                            lnputs, 
                                                            targets, 
                                                            test_size=0.3)

    #build the network architecture
    #우리는 순방향 모델을 사용할 것이고
    model = keras.Sequential([
        #input layer
        #이전에 inputs는 3차원 배열이었고 -> (num_samples, num_time_steps, num_mfcc)
        #이를 7:3 비율로 나눈 inputs_train 역시 3차원 배열
        # Flatten을 통해 input_shape로 1차원 배열로 변환해야 하는데
        # 앞셔 이야기했듯이 3차원 배열에서 
            # shape[0]: 샘플의 총 개수(Batch Size의 상위 개념).
            # shape[1]: 시계열 데이터의 시간 축 길이(Time Steps / Rows).
            # shape[2]: 각 시점에서의 특징 벡터 크기(Features / Columns).
        #인 것이고
        #Flatten을 통해 shape[1], shape[2]를 곱한 1차원 배열로 변환
        #이 과정에서 mfcc의 형태는 사라지지만 해당 정보를 가지고 있으면서
        #완전 연결층(Dense Layer)에 입력할 수 있는 1차원 배열을 얻음
        #그리고 그 1차원 배열의 원소의 개수는 shape[1] * shape[2] = 258*13 = 3354

        #만약 100개의 곡을 10개의 segment로 나누어 각각의 segment에 대해
        #MFCC를 추출했다면, inputs_train의 shape는 (700, 258, 13)이 될 것이고
        # Flatten을 거친 후의 input_shape는 (700, 3354)가 될 것임 
        # 즉 3354개의 원소를 가진 1차원 배열이 700개 있는 형태
        keras.layers.Flatten(input_shape=(inputs_train.shape[1], inputs_train.shape[2])),
        #1st hidden layer
        #각 list에 대해 다음 은닉층이 512개의 뉴런을 가지도록 설정
        #따라서 가중치 행렬은 (3354, 512) 형태가 될 것임
        
        #이때 relu 활성화 함수를 사용하는데
        #relu 함수는 입력이 0 이하일 때는 0을 출력하고
        #0 초과일 때는 입력 값을 그대로 출력하는 함수로서
        #기존 sigmoid 함수에서 발생할 수 있었던
        # 기울기 소실 문제(vanishing gradient problem)를 완화하는 데 도움을 줌
        keras.layers.Dense(512, activation='relu'),
        #2nd hidden layer
        #가중치 행렬은 (512,256) 형태가 될 것임
        keras.layers.Dense(256, activation='relu'),
        #3rd hidden layer
        #가중치 행렬은 (256,64) 형태가 될 것임
        keras.layers.Dense(64, activation='relu'),

        #output layer
        #장르를 하나로 특정하려면 출력층이 하나여야 하는거 아닌가 싶지만
        #정확히 하나로 특정하는 것이 아니라
        #각 장르에 속할 확률을 출력하는 것임
        #따라서 10개의 장르에 대해 각각 확률을 출력해야 함

        #이때 softmax 활성화 함수를 사용하는데
        #softmax 함수는 다중 클래스 분류 문제에서 주로 사용되며
        #각 클래스에 속할 확률을 출력함
        #원리는 각 뉴런이 뱉은 점수($z$)에 **지수 함수($e$, 자연상수)**를 취함.
        #즉 작은 차이의 출력도 크게 벌려서(잘하는 놈을 확실히 밀어주는 개념)
        #해당 배열을 정규화 하여 다 더했을 때 정확히 1이 되도록 만듦
        keras.layers.Dense(10, activation='softmax')
    ])
    #copile network
    #Adam은 확률적 경사 하강법(SGD)을 기반으로 한 최적화 알고리즘으로서
    #학습 속도를 자동으로 조절하여 빠르고 안정적인 수렴을 도와줌
    #학습률(learning rate)은 모델이 학습하는 속도를 결정하는 하이퍼파라미터로서
    #너무 크면 최적의 해를 지나칠 수 있고, 너무 작으면 학습이 너무 느려질 수 있음
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    #우리가 지정한 최적화 방법으로 모델을 컴파일 할건데
    #손실 함수로는 sparse_categorical_crossentropy를 사용할 것임
    #이는 다중 클래스 분류 문제에서 사용되는 손실 함수로서
    #모델이 낸 확률(Softmax 결과)과 실제 정답(0~9 정수) 사이의 오차를 계산함.
    #틀릴수록 이 점수가 커지고, 모델은 이 점수를 낮추기 위해 가중치를 보정함.
    #metrics로는 accuracy를 사용할 것임
    #이는 모델의 예측이 실제 정답과 얼마나 일치하는지를 백분율로 나타낸 지표임
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    #summary는 모델의 구조를 요약해서 보여주는 함수
    #우리가 설계한 모델의 각 층(layer)의 이름, 출력 형태(output shape),
    #파라미터 수(parameters)를 표 형식으로 출력함
    model.summary()

    #train network

    history = model.fit(inputs_train, 
              targets_train,
              validation_data=(inputs_test, targets_test),
              epochs=50,
              #batch는 모델의 학습효율에 대한 개념인데
              #stchastic batch는 하나의 샘플을 학습한 후에 가중치를 수정하는 방식이고
              #full batch 는 전체 샘플을 학습한 후에 가중치를 수정하는 방식임
              #mini-batch는 그 중간 개념으로서 우리가 정한 개수만큼 학습하고 가중치를 수정함

              #각각
              #stochastic batch는 샘플 하나하나의 결과에 예민하게 반응하여
              #학습이 불안정할 수 있지만, 빠르게 수렴하는 경향이 있음
              #full batch는 전체 데이터를 고려하기 때문에 안정적이지만
              #학습 속도가 느릴 수 있음
              #mini-batch는 이 둘의 장점을 절충한 방법으로서
              #적절한 batch size를 선택하는 것이 중요함
              batch_size=32)
    
#돌려보면 알겠지만 testing accuracy가 0.5 정도로 나오고
#loss가 2.0 정도로 나오는 것을 볼 수 있음
#이는 overfitting이 발생한 것으로서, training accuracy는 0.95 정도로 매우 높지만 validation accuracy는 0.59 정도로 낮으며, validation loss가 상승하는 전형적인 과적합 현상이 나타남

# 과적합(Overfitting) 분석:
# 현상: Training Accuracy(95%) 대비 Validation Accuracy(59%)가 현저히 낮으며, Validation Loss가 상승하는 전형적인 과적합 발생.
# 원인: 학습 데이터셋의 규모(100개 샘플)에 비해 모델의 파라미터 수가 과도하게 많아, 데이터의 일반적인 패턴이 아닌 개별 샘플의 노이즈를 암기함.
# 해결 방향: Dropout, 가중치 규제(Regularization), 데이터 증강 등을 통해 모델의 복잡도를 제어하고 일반화 성능을 높이는 공정이 필요함.
    
    #overfitting 분석을 돕기 위한 시각화 함수에 history 전달
    plot_history(history)