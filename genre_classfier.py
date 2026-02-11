import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras


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
    model = keras.Sequential([
        #input layer
        keras.layers.Flatten(input_shape=(inputs_train.shape[1], inputs_train.shape[2])),
        #hidden layer
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        #output layer
        keras.layers.Dense(10, activation='softmax')
    ])
    #co