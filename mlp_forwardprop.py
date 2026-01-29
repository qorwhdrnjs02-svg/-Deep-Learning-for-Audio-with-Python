import numpy as np
# save activations and derivatives

# implement backpropagation
# implement gradient descent
# implement train
# train our net with some dummy dataset
# make some prediction

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
            w = np.random.randn(layers[i], layers[i+1])
            #numpy 라이브러리에서 메트릭스를 만드는 방식
            #즉 3번 반복하면서 각 레이어에 대해
            #첫번쨰는 3x3 행렬
            #두번째는 3x5 행렬
            #세번쨰는 5x2 행렬을 만들라는 소리
            #다만 numpy 라이브러리에서 채워지는 랜덤한 숫자는 모두 0~1범위
            self.weights.append(w)
            
            #순전파 결과 값을 저장해두기 위해 activations 리스트를 만듬
        activations = []
        for i in range (len(layers)):
            #4번 루프를 돌면서
            a =np.zeros(layers[i])
            #numpy 배열을 총 4번 만드는데
            #list가 [3,3,5,2]니까
            #a = np.zeros(0) = [0., 0., 0.]
            #a = np.zeros(1) = [0., 0., 0.]
            #a = np.zeros(2) = [0., 0., 0., 0., 0.]
            #a = np.zeros(4) = [0., 0.]
            #가 되는 것이고
            activations.append(a)
            #activations = [[0., 0., 0.], ... , [0., 0.]]
            #가 되는 것임
            self.activations = activations

            #이제 activations를 이용해 derivatives 리스트를 만듬
        derivative = []
        for i in range (len(layers)-1):
        # 4개의 레이어에 대해 3개의 통로가 있고 
        #이 말은 3개의 미분식이 필요하다는거니까 -1을 해주는 것
            d = np.zeros((layers[i], layers[i+1]))
            #첫번쨰는 3x3 행렬
            #두번째는 3x5 행렬
            #세번쨰는 5x2 행렬을 만듬
            #행렬을 만드는 방식이 가중치를 만들었던 방식과 동일한데
            #이는 각 가중치에 대한 
            #gradient descent를 적용/저장하기 위함임
            derivative.append(d)
        self.derivative = derivative


     
    def forward_propagate(self, inputs):
        #input 벡터를 받아서
        activations = inputs #임시로 저장한 다음에
        self.activations[0] = inputs

        for i, w in enumerate (self.weights):
            net_inputs = np.dot(activations, w) 
            #루프 돌면서 w랑 행렬곱해소 net_input 뽑아내고

            activations =self._sigmoid(net_inputs) 
            #결과로 나온 net_inputs을 sigmoid 함수에 입력
            #여기서 나온 net_inputs는 당연히 행렬인데
            #activations[0]은 1x3 행렬
            # w = self.weights[0]는 3x3 행렬이니까 
            # 1x3 행렬이 activations[1]에 저장이 되고
            #다시 루프를 돌게 되는 것임
            self.activations[i+1] = activations
            #여기서 i+1을 하는 이유는 
            #a_3 = sigmoid(h_3)인데
            #h_3 = a_2*W_2 이기 떄문임

        return activations #activation 반환
    
    def back_propagate(self, error):

        for i in reversed(range(len(self.derivative))):
        #행렬을 담고 있는 self.derivatd는 행렬을 총 3개 담은 list고
        #이를 reversd로 돈다는 것은 인댁스를 2 ->1->0 으로 돈다는 의미

        # dE/dW_i = (y - a_[1+i]) s'(h_[i+1]) a_i
        # s'(h_[i+1]) = s(h_[i+1])(1-s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1] 
        # 6.training a neural network 내용대로 dE/dW를
        # 두 layer 사이의 오류 정도를 나타내는 delta_[i+1]과 
        # 앞선 앞에서 출력한 데이터 a_i의 곱으로 나타낼 수 있다
      


            #여기서 착각하지 말아야 할 것이 activations가 행벡터고
            # simoid_derivative와 error 역시 각각 행벡타이며
            # *는 각 요소별로 곱하는 Hadamard product를 수행한다
            # 그말인 즉슨 delta도 행벡터라는 뜻
            # current_activations는 말할 것도 없이 행벡터이니
            # current_activations를 transpose한 것과 
            # 행벡터인 delta의 행렬 곲 np.dot은
            # 그 값이 행렬로 나올 것이다 --> (ax1) x (1xb) = (axb)
            activations = self.activations[i+1]
            # activations에 뒤의 데이터를 받고
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T 
                        #--> shape[0]은 기존 list에 담긴 데이터의 개수를 나타내고
                        #--> -1은 네가 알아서 반대편은 몇 차원인지 결정하라는 뜻
                        # T는 transpose 
             
            # 그걸 델타로서 계산 delta = (error x sigmoid의 미분)
            current_activations = self.activations[i]
            # 앞선 데이터를 current_activations로 받고
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivative[i] = np.dot(current_activations_reshaped, delta_reshaped)
            # 앞선 데이터(a_i)와 델타(delta_[i+1])의 곱으로 미분을 계산한다
            error = np.dot(delta, self.weights[i].T)
            #최종적으로 enl
            return error

                ############### 예시 ################
                # 1. 이전 층 데이터 (3개)
                # current_activations = np.array([0.1, 0.2, 0.3])
                # # reshape(3, -1) -> (3, 1)로 세웁니다.
                # current_activations_reshaped = [[0.1], 
                #                                 [0.2], 
                #                                 [0.3]]

                # # 2. 델타 (2개)
                # delta = np.array([0.5, 0.8])
                # # 위에서 분석한 대로 (1, 2)로 눕힙니다.
                # delta_reshaped = [[0.5, 0.8]]

                # # 3. np.dot(세로, 가로) 연산 수행
                # # (3x1) dot (1x2) => (3x2)
                # derivative = np.dot(current_activations_reshaped, delta_reshaped)

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
    