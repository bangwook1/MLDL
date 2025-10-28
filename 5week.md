# Lecture 5  
## 딥러닝 학습의 최적화 (Optimization in Deep Learning)

---

### 1️⃣ 딥러닝 학습의 주요 문제점

딥러닝 학습은 다음과 같은 어려움들을 내포한다:

- **경사소실 문제 (Vanishing Gradient Problem)**  
  → 층이 깊어질수록 역전파 시 기울기가 0에 수렴하여 학습이 진행되지 않음  
  → Sigmoid, tanh 함수에서 자주 발생  

- **초기값 설정 문제 (Initialization Problem)**  
  → 잘못된 초기값은 학습 속도 저하 및 수렴 실패를 초래  

- **과대적합 (Overfitting)**  
  → 훈련 데이터에만 지나치게 적합되어 검증/시험 성능 저하  

- **국지적 최소점 (Local Minima) / 긴 학습시간**

---

### 2️⃣ 경사소실과 활성화 함수

- Sigmoid와 tanh는 입력이 커질수록 미분값이 0으로 수렴 → 경사소실 발생  
- **ReLU 함수**
  $$
  f(x) = \max(0, x)
  $$
  → 0 이상에서 미분값이 1로 일정 → 경사소실 완화  
- 변형 ReLU들:
  - Leaky ReLU: $f(x) = \max(0.01x, x)$  
  - Parametric ReLU: 음수 부분 기울기 α를 학습  
  - ELU: $f(x) = x$ (x>0), $\alpha(e^x - 1)$ (x≤0)

---

### 3️⃣ 가중치 초기화

| 방법 | 수식 | 적용 함수 |
|------|------|------------|
| **Xavier 초기값** | $N(0, \frac{2}{n_{in} + n_{out}})$ | Sigmoid, tanh |
| **He 초기값** | $N(0, \frac{4}{n_{in} + n_{out}})$ | ReLU |

- $n_{in}$ : 이전 층 뉴런 수  
- $n_{out}$ : 다음 층 뉴런 수  
- 층별 분산 조정을 통해 학습 안정화

---

### 4️⃣ 확률적 경사하강법 (Stochastic Gradient Descent, SGD)

$$
w_t := w_{t-1} - \eta \frac{\partial J(w)}{\partial w_{t-1}}
$$

- 데이터를 **임의로 추출하여** 한 번에 하나씩 가중치 갱신  
- 빠르지만 진동이 심하고, 경로가 불안정  
- 이전 경사 정보를 기억하지 않음 → 효율 낮음  

---

### 5️⃣ 모멘텀 (Momentum)

$$
m_t = \beta m_{t-1} + \eta \frac{\partial J(w)}{\partial w_t}
$$
$$
w_t := w_{t-1} - m_t
$$

- 이전 기울기 방향을 일정 비율(β)로 반영  
- 손실함수의 진동 완화, 더 부드러운 수렴  
- 일반적으로 **β = 0.9** 사용  
- “공이 언덕을 내려가듯 관성으로 국지적 최소점을 넘음”

---

### 6️⃣ AdaGrad (Adaptive Gradient)

- 각 변수별로 **누적 제곱 기울기**를 고려해 학습률 조정  
- 자주 업데이트되는 파라미터의 학습률 ↓  
- 학습률 자동 조정 식:
  $$
  s_t = s_{t-1} + \left(\frac{\partial J(w)}{\partial w_t}\right)^2
  $$
  $$
  w_t := w_{t-1} - \frac{\eta}{\sqrt{s_t + \epsilon}} \frac{\partial J(w)}{\partial w_t}
  $$
- 초반 빠른 학습 → 후반 너무 느려지는 단점

---

### 7️⃣ RMSProp (Root Mean Square Propagation)

- AdaGrad의 문제(학습률 급감)를 보완  
- 최근 기울기만 고려 (지수이동평균 사용)
  $$
  s_t = \beta s_{t-1} + (1 - \beta)\left(\frac{\partial J(w)}{\partial w_t}\right)^2
  $$
  $$
  w_t := w_{t-1} - \frac{\eta}{\sqrt{s_t + \epsilon}} \frac{\partial J(w)}{\partial w_t}
  $$
- 일반적으로 **β = 0.9**

---

### 8️⃣ Adam (Adaptive Moment Estimation)

Momentum + RMSProp 결합  
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}
$$
$$
w := w - \eta \frac{\hat{m}_t}{\sqrt{\hat{s}_t} + \epsilon}
$$
- **기본 파라미터**: β₁=0.9, β₂=0.999, ε=10⁻⁸  
- 대부분의 딥러닝 프레임워크에서 기본 최적화 알고리즘으로 사용됨

---

### 9️⃣ 배치 정규화 (Batch Normalization)

입력의 분포를 정규화하여 학습 안정화  

$$
\hat{z}^{(l)} = \frac{z^{(l)} - \mu^{(l)}}{\sqrt{\sigma^{2(l)} + \epsilon}}
$$
$$
\tilde{z}^{(l)} = \gamma \hat{z}^{(l)} + \beta
$$

- 각 층 입력의 평균 0, 분산 1로 조정  
- **효과:**  
  - 경사소실 완화  
  - 초기값 의존성 감소  
  - 과대적합 억제  
  - 단점: 추가 파라미터(γ, β)로 복잡도 증가, 학습시간 증가  

---

### 🔟 하이퍼파라미터 최적화

| 항목 | 설명 |
|------|------|
| **학습률 (η)** | 경사하강 이동 크기 |
| **미니배치 크기** | 한 번의 업데이트에 사용되는 샘플 수 |
| **에포크 수** | 전체 데이터를 학습한 횟수 |
| **최적화 방법** | SGD, Adam 등 |
| **정규화 파라미터** | 과적합 방지 |

#### 탐색 방법
- **Grid Search**: 모든 조합 시도  
- **Random Search**: 랜덤 샘플링으로 효율 향상  
- **Bayesian Optimization**: 과거 결과 기반 탐색  

---

> **요약:**  
> Lecture 5에서는 딥러닝 학습의 효율을 높이기 위한 최적화 기법들을 다룸.  
> 경사하강법 기반 알고리즘(SGD, Momentum, AdaGrad, RMSProp, Adam)을 비교하며,  
> 학습 안정화를 위한 **배치 정규화**와 **하이퍼파라미터 최적화**의 중요성을 강조함.

---
