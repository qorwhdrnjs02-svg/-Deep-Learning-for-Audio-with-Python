import numpy as np

class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
    #생성자 선언
    #self는

    # 인공 신경망 객체 초기화
    # :param num_inputs: 입력층의 뉴런 개수 (입력값의 차원)
    # :param num_hidden: 은닉층 리스트. 각 요소는 해당 층의 뉴런(유닛) 개수
    # :param num_outputs: 출력층의 뉴런 개수 (최종 분류/예측값 개수)
    # 예: [3, 5]인 경우 -> 1차 은닉층(3개 뉴런), 2차 은닉층(5개 뉴런) 생성

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden +[self.num_outputs]
        #layers = [3,3,5,2]

        #initiate random weights
        self.weights = []
        for i in range(len(layers)-1):
            #4개의 층에 대하여 3개의 통로 필요
            w = np.random.rand(layers[i], layers[i+1])
            #numpy 라이브러리에서 메트릭스를 만드는 방식
            #즉 3번 반복하면서 각 레이어에 대해
            #첫번쨰는 3x3 행렬
            #두번째는 3x5 행렬
            #세번쨰는 5x2 행렬을 만들라는 소리
            #다만 numpy 라이브러리에서 채워지는 랜덤한 숫자는 모두 0~1범위
            self.weights.append(w)
     
    def forward_propagate(self, inputs):
        #input 벡터를 받아서
        activations = inputs #임시로 저장한 다음에

        for w in self.weights:
            net_inputs = np.dot(activations, w) #루프 돌면서 w랑 행렬곱해소 net_input 뽑아내고

            activations =self._sigmoid(net_inputs) #결과로 나온 net_input을 sigmoid 함수에 입력

        return activations #activation 반환
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
if __name__ == "__main__":
    #create a MLP
    mlp = MLP()
    #create some Inputs
    inputs = np.random.rand(mlp.num_inputs)
    #numpy에서 N차원 벡터를 생성하는 법

    #perform forward prop
    outputs = mlp.forward_propagate(inputs)

    #print the results
    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))
    