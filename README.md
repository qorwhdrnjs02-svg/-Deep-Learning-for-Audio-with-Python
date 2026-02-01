# -Deep-Learning-for-Audio-with-Python

## 01. Artificial Neuron Implementation
- **Goal:** Understand the basic building block of neural networks.
- **Process:**
  1. Receive multiple inputs.
  2. Apply weights to each input.
  3. Calculate the weighted sum.
  4. Use Sigmoid activation function to squeeze the result between 0 and 1.
- Weighted Sum: $h = \sum w_i x_i$
- Sigmoid: $y = \frac{1}{1 + e^{-h}}$
- **Tools:** Python `math` module.

## 02. Multilayer Perceptron Implementation
- **Goal:** 여러 은닉층을 가진 신경망 구조 설계 및 순전파 구현.
- **Process:**
  1. 입력층, 은닉층 리스트, 출력층을 포함한 전체 레이어 구조 정의.
  2. `numpy.random.randn`을 이용해 각 층 사이의 가중치 행렬 초기화.
  3. `numpy.dot` 행렬 곱셈을 통해 층간 데이터 흐름 구현.
  4. 반복문을 통해 각 층에 Sigmoid 활성화 함수 적용 및 최종 결과 도출.
- **Tools:** Python `NumPy`.

## 02-1. Structure Simulation

본 프로젝트의 동작 원리를 이해하기 위해 2(입력)-2(은닉)-1(출력) 레이어 구조를 기준으로 한 연산 과정을 정리합니다.

### 1. 전제 조건 (Notation)
* **Activations:** $a_0$ (입력층), $a_1$ (은닉층), $a_2$ (출력층)
* **Weights:** $W_0$ (입력-은닉 가중치), $W_1$ (은닉-출력 가중치)
* **Functions:** $\sigma$ (Sigmoid 활성화 함수), $\sigma'$ (Sigmoid 미분)
* **Target:** $y$ (실제 정답)

---

### 2. 순전파 (Forward Propagation)
데이터가 입력층에서 출력층으로 흐르며 예측값 $a_2$를 도출하는 과정입니다.

1.  **은닉층 계산:**
    $$h_1 = a_0 \cdot W_0$$
    $$a_1 = \sigma(h_1)$$
2.  **출력층 계산:**
    $$h_2 = a_1 \cdot W_1$$
    $$a_2 = \sigma(h_2)$$



---

### 3. 역전파 (Backpropagation)
오차를 바탕으로 뒤에서부터 각 층의 기여도(Delta)를 계산하고 가중치를 수정하는 과정입니다.

#### **Step 1: 출력층 (Output Layer)**
1.  **출력층 에러 ($error_2$):** 실제값과 예측값의 차이
    $$error_2 = y - a_2$$
2.  **출력층 델타 ($\delta_2$):** 에러에 출력층 기울기를 적용 (Hadamard product)
    $$\delta_2 = error_2 \odot \sigma'(a_2)$$
3.  **가중치 $W_1$ 수정안 ($Deriv_1$):**
    $$Deriv_1 = a_1^T \cdot \delta_2$$

#### **Step 2: 은닉층 (Hidden Layer)**
1.  **은닉층 에러 ($error_1$):** 출력층 델타가 가중치를 타고 역전파됨
    $$error_1 = \delta_2 \cdot W_1^T$$
2.  **은닉층 델타 ($\delta_1$):** 배달된 에러에 은닉층 기울기를 적용 (Hadamard product)
    $$\delta_1 = error_1 \odot \sigma'(a_1)$$
3.  **가중치 $W_0$ 수정안 ($Deriv_0$):**
    $$Deriv_0 = a_0^T \cdot \delta_1$$



---

### 4. 가중치 업데이트 (Weight Update)
계산된 미분값($Deriv$)과 학습률($\eta$)을 사용하여 가중치를 실제로 수정합니다.

* $W_1 = W_1 + \eta \cdot Deriv_1$
* $W_0 = W_0 + \eta \cdot Deriv_0$

---

### 💡 핵심 원리 요약
* **$\delta$ (Delta):** 각 층의 뉴런이 결과에 대해 책임져야 할 **오차의 본체**입니다. 항상 `error ⊙ f'(a)`의 일관된 형태로 계산됩니다.
* **$\cdot$ (Dot Product):** 에러 신호를 앞 층으로 **전달**하거나, 가중치 전체의 **수정 지도**를 그릴 때 사용합니다.
* **$\odot$ (Hadamard Product):** 해당 층의 활성화 함수 특성(기울기)을 에러에 **필터링**할 때 사용합니다.

### 🔢 수치 기반 시뮬레이션 (Numerical Example)

2-2-1 구조에서 실제 숫자가 어떻게 계산되고 전달되는지 단계별로 살펴봅니다.

#### **1. 순전파 (Forward)**
* **Input ($a_0$):** $[1.0, 2.0]$
* **Weights ($W_1$):** 모두 $0.5$로 가정 (은닉-출력 가중치)
* **Target ($y$):** $1.0$ (목표 정답)
* **Output ($a_2$):** $0.68$ (예측값), **Hidden ($a_1$):** $[0.6, 0.7]$

#### **2. 역전파 (Backward) - 출력층**
1.  **$error_2$ 계산:**
    $$error_2 = y - a_2 = 1.0 - 0.68 = \mathbf{0.32}$$
2.  **$\delta_2$ 계산 (오차 $\odot$ 기울기):**
    * $\sigma'(a_2) = 0.68 \times (1 - 0.68) \approx 0.22$
    $$\delta_2 = error_2 \odot \sigma'(a_2) = 0.32 \times 0.22 = \mathbf{0.07}$$
3.  **$Deriv_1$ 작성 (이전 층 출력 $a_1^T$와 델타 $\delta_2$의 내적):**
   
$$
\text{Deriv}_1 = \left[ \begin{matrix} 0.6 \\\\ 0.7 \end{matrix} \right] \cdot [0.07] = \left[ \begin{matrix} 0.042 \\\\ 0.049 \end{matrix} \right]
$$

#### **3. 역전파 (Backward) - 은닉층**
1.  **$error_1$ 전달 (에러 릴레이):**
    * $\delta_2$가 가중치 $W_1$을 타고 역행
    $$error_1 = \delta_2 \cdot W_1^T = [0.07] \cdot [0.5, 0.5] = \mathbf{[0.035, 0.035]}$$
2.  **$\delta_1$ 계산 (배달된 에러 $\odot$ 은닉층 기울기):**
    * $\sigma'(a_1) = [0.6(1-0.6), \; 0.7(1-0.7)] = [0.24, 0.21]$
    $$\delta_1 = error_1 \odot \sigma'(a_1) = [0.035, 0.035] \odot [0.24, 0.21] = \mathbf{[0.0084, 0.00735]}$$
3.  **$Deriv_0$ 작성 (입력값 $a_0^T$와 델타 $\delta_1$의 내적):**
   
$$
Deriv_0 = \begin{bmatrix} 1.0 \\\\ 2.0 \end{bmatrix} \cdot [0.0084, 0.00735]
$$

$$
Deriv_0 = \mathbf{\begin{bmatrix} 0.0084 & 0.00735 \\\\ 0.0168 & 0.0147 \end{bmatrix}}
$$

## 03-2. Training & Learning Implementation
- **Goal:** 데이터 세트를 반복 학습시켜 가중치를 실제로 수정함.
- **Process:**
  1. **데이터 공급:** `zip`을 이용해 `inputs`와 `targets`를 한 쌍의 세트로 묶어 꺼냄.
  2. **반복 학습:** `epochs` 횟수만큼 전체 데이터를 반복해서 수행함.
  3. **가중치 수정:** `gradient_descent`를 통해 계산된 미분값과 학습률을 연산하여 가중치를 업데이트함.

### 💡 학습 및 오차 분석 논리
- **데이터 추출 방식:** `zip`은 문제와 정답을 하나로 묶어주고, 루프를 돌며 각각의 세트를 꺼내 함수 인자로 사용함.
- **에러 누적 및 평균화:** - 각 루프마다 발생하는 `mse` 오차를 `sum_error`에 쌓음.
  - 한 epoch 루프가 끝날 때, 누적된 값을 **사용한 데이터 세트의 수(`len(inputs)`)**로 나눠야 데이터당 평균 오차를 확인할 수 있음.

## 04. Final Simulation Result & Test
- **Task:** 덧셈 연산 학습 ($x_1 + x_2$)
- **Dataset:** `random()/2`를 사용해 합이 1을 넘지 않는 1,000개의 데이터 생성.
- **Training:** 50 에포크 동안 총 50,000번의 학습 수행.
