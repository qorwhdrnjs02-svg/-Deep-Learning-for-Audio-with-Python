import numpy as np
from random import random
import tensorflow as tf
# 학습 데이터와 별개로, 모델의 진짜 성능을 검증할 '시험용 데이터'를 떼어놓기 위한 라이브러리
from sklearn.model_selection import train_test_split

#array([[0.1, 0.2], [0.2, 0.2]])
#array([[0.3], [0.4]])

def generate_dataset(num_samples, test_size):
    #2000개의 더하기 데이터 샘플 생성
    # 0~0.5 사이의 두 값을 요소로 갖는 벡터 x 와 두 요소의 더하기 결과 y 
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0]+ i[1]]for i in x])
    # 우리가 모델을 학습하는 동안 함수가 확인한 임의의 x에 대해서 정답을 잘 내지만
    # 새로운 문제를 접했을 때 모델이 정확한 target 값을 내는지 확인하는 과정이 필요
    # 따라서 1-test_size : test_size의 비율로 학습용 : 테스트용 데이터 샘플을 나눔
    # ex) 시험공부용 70 : 시험문제 30 으로 나눈다고 생각할 수 있음

    #train_test_split 함수는 애초에 변수 개수에 따라 변수1_train, 변수1_test 를 뱉어내게 설계됨
    #따라서 이렇게 여러개의 변수를 한번에 정의할 수 있고
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size)

    #해당 변수들을 반환하고
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    # 우리가 gen 함수에 인자로 준 샘플 개수와, 데스트 사이즈 비율에 따라
    # 각 변수들이 정의되고
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
    # 10을 0.2의 비율로 테스트 샘플을 나눴으니 2개의 테스트가 나올 것
    # print("x_test: \n {}".format(x_test))
    # print("y_test: \n {}".format(y_test))

#build model: 2 -> 5 -> 1 의 모델을 구축한다면
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_dim = 2, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#compile model
optimiser = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimiser, loss="MSE")

#train model
model.fit(x_train, y_train, epochs=100)

#evaluate model
#당연한 이야기자만 loss 값이 train model과 많은 차이가 나면
# 제대로 학습이 진행되지 않았다는 거니까 문제가 있음 
print("Model evaluation:")
model.evaluate(x_test,y_test, verbose=1)

#make predictions
data = np.array([[0.1, 0.2], [0.2, 0.2]])
prediction = model.predict(data)

print("Some predcition: ")
for d, p in zip(data, prediction):
    print("{} + {} = {}".format(d[0], d[1], p[0]))

