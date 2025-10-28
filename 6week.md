# Lecture 6  
## 합성곱신경망의 응용 (Applications of CNN)

---

### 1️⃣ CNN의 발전 과정
| 모델 | 주요 특징 |
|------|-----------|
| **LeNet-5 (1998)** | 최초의 CNN 구조, 손글씨 숫자 인식 |
| **AlexNet (2012)** | ReLU, 드롭아웃, GPU 병렬연산, LRN 적용 |
| **VGGNet (2014)** | 3×3 필터 반복, 단순·깊은 구조 |
| **GoogLeNet (2014)** | 인셉션 모듈로 효율적 연산 |
| **ResNet (2015)** | 스킵연결(skip connection)로 경사소실 해결 |

---

### 2️⃣ AlexNet
- 입력: 227×227×3 컬러 이미지  
- 5개의 합성곱층 + 3개의 완전연결층  
- 활성화: **ReLU**, 과적합 방지를 위한 **드롭아웃(50%)**  
- 학습률 0.9, 배치크기 128, 모멘텀 SGD 사용  
- GPU 병렬 연산으로 속도 향상  
- **지역반응정규화(LRN)** 적용 → 과도한 활성 억제  

#### 가중치 구성
- 합성곱층 약 6%, 완전연결층 94%  
- 총 62,369,152개 파라미터  

---

### 3️⃣ 지역반응정규화 (Local Response Normalization)
- 한 위치에서 뉴런이 과도하게 활성화될 경우,  
  주변 뉴런의 출력을 억제하여 과적합 방지  
$$
b_{x,y}^i = a_{x,y}^i / \left(k + \alpha \sum_{j=max(0, i-n/2)}^{min(N-1, i+n/2)} (a_{x,y}^j)^2 \right)^\beta
$$

---

### 4️⃣ 객체 검출 (Object Detection)
> 단순한 “무엇인지 분류”를 넘어 “어디에 있는지 위치를 찾는 것”

#### 주요 개념
- **분류(Classification)**: 객체의 종류 예측  
- **지역화(Localization)**: 객체의 위치(경계상자) 찾기  
- **객체 검출(Object Detection)**: 둘을 동시에 수행  
- **분할(Segmentation)**:
  - *Semantic*: 같은 범주의 객체를 하나로 구분  
  - *Instance*: 같은 범주라도 개별 객체 구분  

---

### 5️⃣ 객체 검출 데이터셋
| 데이터셋 | 특징 |
|-----------|------|
| **ImageNet** | 200개 객체, 이미지 48만개, 객체 53만개 |
| **PASCAL VOC** | 20개 범주, 이미지당 2.4개 객체 |
| **MS COCO** | 80개 이상 범주, 12만장 이미지, 90만 객체 |

---

### 6️⃣ 객체 검출 알고리즘
| 구분 | 예시 | 설명 |
|------|------|------|
| **1단계(One-stage)** | YOLO, SSD | 전체 이미지를 한 번에 처리 |
| **2단계(Two-stage)** | R-CNN, Fast R-CNN, Faster R-CNN | 영역을 제안한 뒤 분류 수행 |

---

### 7️⃣ R-CNN 계열
#### (1) R-CNN (Region-CNN)
- *Selective Search*로 약 2000개의 후보영역 생성  
- 각 영역에 CNN 적용 → 특징 추출  
- SVM으로 분류, 회귀로 박스 보정  
- 정확도 높지만 속도 느림 (한 이미지당 47초)

#### (2) Fast R-CNN
- 전체 이미지 한 번만 CNN 통과  
- ROI Pooling으로 영역 특징 추출  
- Softmax 분류 + 회귀 동시 학습  

#### (3) Faster R-CNN
- **Region Proposal Network (RPN)** 추가  
  → 후보영역 자동 생성  
- 속도·정확도 모두 향상

---

### 8️⃣ YOLO (You Only Look Once)
- 이미지를 **격자(grid)** 로 나누어  
  각 셀이 객체 존재 확률과 bounding box 예측  
- 실시간 탐지 가능 (속도 ↑ 정확도 유지)  
- 회귀문제로 처리  
$$
P(\text{object}) \times P(\text{class} | \text{object})
$$

---

### 9️⃣ 영상 분할 (Image Segmentation)

#### (1) FCN (Fully Convolutional Network)
- 완전합성곱 구조로 픽셀 단위 분류  
- 인코더(합성곱) → 디코더(업샘플링) 구조  
- 업샘플링으로 입력과 동일 크기 복원  
- FCN-32s, FCN-16s, FCN-8s 구조 존재

#### (2) U-Net
- FCN 기반, 의학영상에서 활용  
- 인코더-디코더 대칭 구조(U자형)  
- 스킵연결(skip connection)로 세밀한 경계 복원  

#### (3) Mask R-CNN
- Faster R-CNN에 **Segmentation mask** 추가  
- bounding box + 분할 결과 동시 예측  

---

### 🔟 얼굴 인식 (Face Recognition)

#### (1) 얼굴식별 (Identification)
- 입력된 얼굴이 등록된 사람과 **일치하는지 판별**

#### (2) 얼굴검증 (Verification)
- 두 얼굴 이미지의 **특징맵 거리** 비교  
  → 작으면 동일인, 크면 다른 사람  
- **사전처리(preprocessing)** 중요 (정면화 등)

---

### 11️⃣ 전이학습 (Transfer Learning)
- 기존 학습된 CNN을 새 데이터에 재활용  
| 데이터 조건 | 방법 |
|--------------|------|
| 데이터 적음 | 기존 모델 그대로 사용 |
| 데이터 많음 | 상위층 유지, 하위층 추가 학습 |
| 데이터 분포 상이 | 더 많은 층 재학습 |

---

> **요약:**  
> Lecture 6은 CNN의 응용으로 객체 검출, 분할, 얼굴인식, 전이학습을 다룸.  
> 대표모델인 **R-CNN 계열과 YOLO**의 차이를 중심으로  
> 실제 이미지 인식 문제에서의 **속도-정확도-효율성의 균형**을 이해하는 것이 핵심이다.

---
