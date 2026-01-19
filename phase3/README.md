# Phase 3: Two-Track Hybrid Link Prediction

공급망 네트워크에서 **동적 예측(Temporal)** + **구조적 예측(Structural)** 을 결합한 링크 예측 시스템

---

## 🎯 목표

1. **Track A (SC-TGN)**: 시계열 동적 예측
   - 시간의 흐름에 따른 거래 지속성, 주기성, 추세 반영
   - Memory 기반으로 과거 충격을 기억하여 미래 단절 예측

2. **Track B (GraphSEAL)**: 구조적 패턴 예측
   - 잠재적 연결 가능성 탐색 (Sub-graph pattern)
   - UKGE 적용: TIS 기반 신뢰도 점수를 반영해 학습

3. **Ensemble**: 두 모델의 예측 확률(Logit)을 가중 합산하여 최종 Score 산출

4. **Metric**: **Recall@K** (상위 K개 추천 중 실제 거래처가 포함될 확률)

---

## 📂 파일 구조

```
Phase 3 파일들:

src/
├── temporal_graph_builder.py    # 시계열 그래프 데이터 빌더
├── sc_tgn.py                     # Track A: SC-TGN 모델
├── graphseal.py                  # Track B: GraphSEAL + UKGE + Ensemble
└── hybrid_trainer.py             # Hybrid 학습 트레이너

main_phase3_hybrid.py             # 전체 학습 (100 epochs) ⭐
quick_test_phase3.py              # 빠른 테스트 (5 epochs)
requirements_phase3.txt           # Phase 3 의존성
PHASE3_README.md                  # 이 파일
```

---

## 🔧 설치

```bash
pip install -r requirements_phase3.txt
```

**주요 의존성:**
- PyTorch >= 2.0
- PyTorch Geometric
- NumPy, Pandas, Scipy

---

## 🚀 실행 방법

### 1️⃣ 빠른 테스트 (5 epochs, 작은 모델)

```bash
python quick_test_phase3.py
```

**설정 (QuickConfig):**
- Epochs: 5
- TGN Memory Dim: 64 (작게)
- GraphSEAL Hidden Dim: 64 (작게)
- Batch Size: 512
- 네거티브 비율: 0.5 (낮춤)

**예상 실행 시간:** 5-10분 (CPU), 2-5분 (GPU)

---

### 2️⃣ 전체 학습 (100 epochs, 큰 모델)

```bash
python main_phase3_hybrid.py
```

**설정 (Config):**
- Epochs: 100
- TGN Memory Dim: 128
- TGN Embedding Dim: 64
- GraphSEAL Hidden Dim: 128
- Batch Size: 1024
- 네거티브 비율: 1.0
- Early Stopping: 15 epochs

**예상 실행 시간:** 1-2시간 (CPU), 20-40분 (GPU)

---

## 📊 입력 데이터

Phase 3는 다음 파일들을 사용합니다:

### Phase 1 출력
- `data/processed/disentangled_recipes.pkl` (사용 X, Phase 2에서 사용됨)

### Phase 2 출력
- `data/processed/node_embeddings_static.pt` ⭐ (Static Embeddings)
- `data/processed/train_edges.npy` (Train 엣지 인덱스)
- `data/processed/test_edges.npy` (Test 엣지 인덱스)
- `data/processed/X_feature_matrix.npy` (노드 피처)
- `data/processed/tis_score_normalized.npy` (TIS 점수)

### Phase 3 고유 데이터
- `data/raw/posco_network_2020.csv` 🕐 시계열 네트워크
- `data/raw/posco_network_2021.csv`
- `data/raw/posco_network_2022.csv`
- `data/raw/posco_network_2023.csv`
- `data/raw/firm_to_idx_model2.csv` (기업 ID 매핑)

**시계열 네트워크 CSV 컬럼:**
- `Unnamed: 0` 또는 `source`: Source 기업 ID
- `Unnamed: 1` 또는 `target`: Target 기업 ID
- `transaction_amount` (옵션): 거래액
- `frequency` (옵션): 거래 빈도

**기업 ID 매핑 CSV 컬럼:**
- `Unnamed: 0` 또는 `firm_id`: 기업 ID
- `idx`: 인덱스 (0부터 시작)

---

## 🧠 모델 아키텍처

### Track A: SC-TGN (Supply Chain Temporal Graph Network)

```
[입력] 시계열 이벤트 스트림 (timestamp, src, dst, edge_feat)
    ↓
[Memory Module] 각 노드의 과거 상호작용 기억
    ↓
[Time Encoder] 시간 간격을 피처로 변환
    ↓
[Message Aggregator] 이웃 메시지 집계
    ↓
[Memory Updater (GRU)] 메모리 갱신
    ↓
[Embedding Layer] 최종 임베딩 생성
    ↓
[출력] 링크 예측 Logits (내적)
```

**핵심:**
- **Memory**: 각 노드가 과거 상호작용을 GRU로 기억
- **시간 인코딩**: 시간 간격을 Cosine 함수로 인코딩
- **메시지 패싱**: 이벤트 발생 시 src↔dst 간 메시지 교환

---

### Track B: GraphSEAL (Structural Pattern + UKGE)

```
[입력] Static Embeddings (Phase 2 출력) + Sub-graph
    ↓
[Subgraph Encoder] k-hop 이웃 정보 집계
    ↓
[MLP Link Predictor] 임베딩 결합 → Logits
    ↓
[UKGE Confidence Scorer] TIS 기반 신뢰도 점수 (0~1)
    ↓
[출력] 링크 예측 Logits + Confidence
```

**핵심:**
- **UKGE (Uncertain Knowledge Graph Embedding)**: TIS가 낮은 엣지는 신뢰도를 낮춰 학습
- **Sub-graph Pattern**: k-hop 이웃 정보를 활용해 local structure 반영

---

### Ensemble: Hybrid Link Predictor

```
Track A Logits (TGN)     Track B Logits (GraphSEAL)
        ↓                           ↓
        α * Logit_A    +   (1-α) * Logit_B
                    ↓
            Weighted Sum
                    ↓
        × Confidence (UKGE)
                    ↓
        Final Logits → Sigmoid → Score
```

**α (가중치)**:
- 초기값: 0.5
- **학습 가능한 파라미터** (Gradient Descent로 최적화)
- Sigmoid로 0~1 범위로 제한

---

## 📈 학습 프로세스

### 1. 데이터 준비

```python
# 시계열 이벤트 로드 (2020-2023)
temporal_data = TemporalGraphBuilder.build_temporal_data()

# Train/Val/Test 분할
# - Train: 2020-2022 (80%)
# - Val: 2020-2022 (20%)
# - Test: 2023 (전체)

# 네거티브 샘플링 (1:1 비율)
# - Positive 이벤트에 대해 랜덤 네거티브 생성
# - Self-loop, Positive 중복 제거
```

### 2. 학습 루프

```python
for epoch in range(epochs):
    # TGN 메모리 초기화
    model.tgn.reset_memory()
    
    for batch in train_loader:
        # Forward (Hybrid)
        logits, outputs = model(...)
        
        # Loss (BCE)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # TGN 메모리 업데이트 (중요!)
        model.tgn.update_memory_with_batch(...)
```

### 3. 평가 메트릭

**Recall@K:**
```
Recall@K = (상위 K개 예측 중 실제 Positive 개수) / (전체 Positive 개수)
```

**예시:**
- 전체 Positive: 100개
- 상위 50개 예측 중 Positive: 30개
- **Recall@50 = 30 / 100 = 0.30**

---

## 📊 하이퍼파라미터

### Track A (SC-TGN)

| 파라미터 | 전체 학습 | 빠른 테스트 | 설명 |
|---------|----------|-----------|------|
| `MEMORY_DIM` | 128 | 64 | 메모리 벡터 차원 |
| `TIME_DIM` | 32 | 16 | 시간 인코딩 차원 |
| `MESSAGE_DIM` | 128 | 64 | 메시지 차원 |
| `EMBEDDING_DIM` | 64 | 32 | 최종 임베딩 차원 |

### Track B (GraphSEAL)

| 파라미터 | 전체 학습 | 빠른 테스트 | 설명 |
|---------|----------|-----------|------|
| `HIDDEN_DIM` | 128 | 64 | 은닉층 차원 |
| `NUM_HOPS` | 2 | 1 | Sub-graph k-hop |
| `USE_UKGE` | True | True | UKGE 사용 여부 |

### 학습

| 파라미터 | 전체 학습 | 빠른 테스트 | 설명 |
|---------|----------|-----------|------|
| `EPOCHS` | 100 | 5 | 학습 에폭 수 |
| `BATCH_SIZE` | 1024 | 512 | 배치 크기 |
| `LEARNING_RATE` | 0.001 | 0.001 | 학습률 |
| `WEIGHT_DECAY` | 1e-5 | 1e-5 | L2 정규화 |
| `EARLY_STOPPING` | 15 | 3 | Early Stopping |
| `NEG_RATIO` | 1.0 | 0.5 | 네거티브 비율 |

---

## 📁 출력 파일

### 전체 학습 (`main_phase3_hybrid.py`)

```
results/
├── hybrid_model_best.pt           # 최고 성능 모델 가중치
└── phase3_metrics.npz             # 학습/평가 메트릭
    ├── test_metrics                  # Test Recall@K, Loss
    ├── train_losses                  # Epoch별 Train Loss
    ├── val_losses                    # Epoch별 Val Loss
    └── val_recalls                   # Epoch별 Val Recall@50
```

### 빠른 테스트 (`quick_test_phase3.py`)

```
results/quick_test/
├── hybrid_model_quick.pt          # 빠른 테스트 모델
└── phase3_metrics_quick.npz       # 빠른 테스트 메트릭
```

---

## 🔍 결과 분석

### 메트릭 로드 및 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# 메트릭 로드
data = np.load('results/phase3_metrics.npz', allow_pickle=True)

# Test Recall@K
test_metrics = data['test_metrics'].item()
print("Test Recall@K:")
for k in [10, 50, 100, 500, 1000]:
    print(f"  Recall@{k}: {test_metrics[f'recall@{k}']:.4f}")

# 학습 곡선
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(data['train_losses'], label='Train Loss')
plt.plot(data['val_losses'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(data['val_recalls'], label='Val Recall@50')
plt.xlabel('Epoch')
plt.ylabel('Recall@50')
plt.legend()
plt.title('Validation Recall@50')

plt.tight_layout()
plt.savefig('results/phase3_curves.png')
```

---

## 📈 종합 평가 (Comprehensive Evaluation)

### 벤치마크 비교

GNN 모델과 고전적 휴리스틱 알고리즘 비교:

```bash
python evaluate_phase3_comprehensive.py
```

**비교 대상:**

1. **PA (Preferential Attachment)**
   - Score(u, v) = degree(u) × degree(v)
   - 차수가 높은 노드끼리 연결될 확률이 높다

2. **RA (Resource Allocation)**
   - Score(u, v) = Σ_{z ∈ common} 1 / degree(z)
   - 공통 이웃이 많을수록, 이웃의 차수가 작을수록 높은 점수

3. **JC (Jaccard Coefficient)**
   - Score(u, v) = |common| / |union|
   - 두 노드의 이웃 집합 유사도

**평가 메트릭:**

| 메트릭 | 설명 | 목적 |
|--------|------|------|
| **Recall@K** | 상위 K개 예측 중 실제 Positive 비율 | 잠재 거래 발굴 능력 |
| **MRR** | Positive의 평균 역순위 | 랭킹 정확도 |
| **RMSE** | 예측값과 실제값(TIS 반영) 오차 | 리스크 학습 검증 |

**예상 출력:**

```
================================================================================
Model           Recall@10    Recall@50    Recall@100   MRR         
================================================================================
GNN (Ours)      0.3245       0.5812       0.7234       0.4123      
PA              0.1234       0.2456       0.3567       0.2012      
RA              0.1567       0.2890       0.4123       0.2345      
JC              0.1423       0.2712       0.3945       0.2198      
================================================================================

📊 RMSE (Risk-aware Prediction Error):
----------------------------------------
  - RMSE (Overall): 0.2145
  - RMSE (Positive): 0.1934
  - RMSE (Negative): 0.0234
  - RMSE (TIS-aware): 0.1867
  - RMSE (Confidence-weighted): 0.1789
----------------------------------------
```

---

### 견고성 테스트 (Robustness Test) - 옵션

Negative 비율을 1:1 → 1:4로 증가시키며 모델 성능 평가:

```python
# evaluate_phase3_comprehensive.py 실행 후
# "견고성 테스트를 실행하시겠습니까?" 질문에 'y' 입력
```

**테스트 시나리오:**
- Neg Ratio 1:1 (기본)
- Neg Ratio 1:2 (Negative 2배)
- Neg Ratio 1:3 (Negative 3배)
- Neg Ratio 1:4 (Negative 4배)

**목적:** 노이즈(Negative) 속에서도 진짜 거래를 잘 찾아내는지 검증

**출력:**
- `results/robustness_test.png` (성능 변화 그래프)
- 콘솔에 테이블 출력

```
견고성 테스트 요약:
----------------------------------------------------------------------
Neg Ratio    Recall@10    Recall@50    MRR         
----------------------------------------------------------------------
1:1.0        0.3245       0.5812       0.4123      
1:2.0        0.2834       0.5234       0.3789      
1:3.0        0.2512       0.4856       0.3456      
1:4.0        0.2298       0.4523       0.3201      
----------------------------------------------------------------------
```

---

## 🐛 디버깅 팁

### 1. 메모리 부족 (OOM)

```python
# Config에서 조정:
BATCH_SIZE = 512  # 줄이기
TGN_MEMORY_DIM = 64  # 줄이기
GRAPHSEAL_HIDDEN_DIM = 64  # 줄이기
```

### 2. 학습이 너무 느림

```python
# 빠른 테스트로 먼저 확인:
python quick_test_phase3.py

# 또는:
EPOCHS = 10  # 줄이기
NEG_RATIO = 0.5  # 낮추기
```

### 3. Recall@K가 너무 낮음

- **원인 1**: 네거티브 샘플이 너무 쉬움 → 더 어려운 네거티브 필요
- **원인 2**: 모델이 너무 작음 → `HIDDEN_DIM`, `MEMORY_DIM` 키우기
- **원인 3**: Early Stopping이 너무 빠름 → `PATIENCE` 늘리기

### 4. TIS 파일이 없음

```bash
# Phase 2를 먼저 실행:
python main_phase2.py
```

---

## 📚 참고 문헌

1. **TGN (Temporal Graph Networks)**
   - Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs", ICLR 2020

2. **UKGE (Uncertain Knowledge Graph Embedding)**
   - Chen et al., "UKGE: Learning Knowledge Graph Embeddings with Uncertainty", AAAI 2019

3. **GraphSAINT**
   - Zeng et al., "GraphSAINT: Graph Sampling Based Inductive Learning Method", ICLR 2020

---

## ✅ 체크리스트

실행 전 확인:

- [ ] Phase 1 완료 (`data/processed/disentangled_recipes.pkl` 존재)
- [ ] Phase 2 완료 (`data/processed/node_embeddings_static.pt` 존재)
- [ ] 시계열 네트워크 파일 존재 (`data/raw/posco_network_20*.csv`)
- [ ] PyTorch Geometric 설치 완료
- [ ] GPU 사용 가능 (선택사항, CPU도 가능)

---

## 🎯 다음 단계

Phase 3 완료 후:

1. **Recall@K 분석**: 어떤 K에서 성능이 좋은가?
2. **Track별 기여도 분석**: α 값 확인, Track A vs Track B 성능 비교
3. **UKGE 효과 분석**: TIS 높은 엣지 vs 낮은 엣지 예측 정확도
4. **시계열 패턴 분석**: 어떤 시기에 예측이 잘/못 되는가?
5. **실제 추천 시스템 구축**: 상위 K개 추천을 실제 비즈니스에 적용

---

**작성일**: 2026-01-19  
**버전**: 1.0  
**작성자**: GNN Pipeline (Phase 3)
