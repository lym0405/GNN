# GraphSEAL Fast Training Guide

**생성 날짜:** 2024-01-20  
**목적:** 시간 절약을 위해 GraphSEAL만 사용하여 빠르게 학습

---

## 🚀 주요 변경사항

### 1. **모델 규모 증가**
기존 Hybrid 모델보다 성능 유지를 위해 GraphSEAL 단일 모델의 규모를 대폭 증가:

| 파라미터 | 기존 | 현재 | 변경 |
|---------|------|------|------|
| Embedding Dim | 128 | 256 | 2배 ↑ |
| Hidden Dim | 256 | 512 | 2배 ↑ |
| Num Layers | 3 | 4 | +1 |
| Num Hops | 2 | 3 | +1 |

### 2. **학습 설정**
- ✅ **에폭:** 30 (빠른 학습)
- ✅ **배치 크기:** 256 (메모리 허용 범위 내 최대)
- ✅ **학습률:** 0.001
- ✅ **Negative Sampling:** 5:1 비율

### 3. **제거된 부분**
- ❌ SC-TGN (시간이 오래 걸림)
- ❌ Hybrid Training (복잡도 제거)
- ❌ TIS Loss 가중치 (단순화)

---

## 📋 필요 입력 파일

```bash
data/raw/
├── H_csr_model2.npz                    # Supply network
└── firm_to_idx_model2.csv              # Firm ID mapping

data/processed/
├── train_edges.npy                     # Train edges (Phase 2)
├── test_edges.npy                      # Test edges (Phase 2)
└── node_embeddings_static.pt           # Node embeddings (Phase 2)
```

---

## 🏃 실행 방법

```bash
# 1. 프로젝트 루트로 이동
cd /Users/iyulim/Desktop/나이스/GNN

# 2. 필요 파일 확인
ls -la data/processed/
# train_edges.npy, test_edges.npy, node_embeddings_static.pt가 있어야 함

# 3. GraphSEAL Fast Training 실행
python phase3/train_graphseal_fast.py

# GPU가 있으면 자동으로 사용됨
# CPU만 있으면 CPU로 학습 (조금 느림)
```

---

## 📊 예상 출력

### 학습 진행 로그:
```
================================================================================
GraphSEAL Fast Training - Starting
================================================================================
Device: cuda
Epochs: 30
Batch size: 256
Model: Embedding=256, Hidden=512, Layers=4

Loading data...
Loaded H matrix: shape (N, N), E edges
Loaded node embeddings: (N, D)
Train edges: X, Test edges: Y

Generating negative samples...
Generated Z negative samples

Creating dataloaders...
Train batches: A, Test batches: B

Initializing GraphSEAL model...
Model parameters: ~2,000,000

================================================================================
Starting training...
================================================================================
Epoch 1/30 (15.2s) | Train Loss: 0.4532 | Test Loss: 0.4123 | AUC: 0.7845 | AP: 0.7623 | Acc: 0.7234
  → New best AUC: 0.7845
Epoch 2/30 (14.8s) | Train Loss: 0.3821 | Test Loss: 0.3756 | AUC: 0.8123 | AP: 0.7934 | Acc: 0.7456
  → New best AUC: 0.8123
...
Epoch 30/30 (14.5s) | Train Loss: 0.2134 | Test Loss: 0.2876 | AUC: 0.8956 | AP: 0.8734 | Acc: 0.8234

================================================================================
Training completed! Best AUC: 0.8956 at epoch 28
================================================================================

Saving results...
Saved model to phase3/output/graphseal_fast/graphseal_model.pt
Saved embeddings to phase3/output/graphseal_fast/node_embeddings_graphseal.pt
```

---

## 📁 출력 파일

학습 완료 후 생성되는 파일:

```bash
phase3/output/graphseal_fast/
├── graphseal_model.pt                  # 학습된 GraphSEAL 모델
├── node_embeddings_graphseal.pt        # 최종 노드 임베딩
└── logs/
    └── train_20260120_HHMMSS.log      # 학습 로그
```

---

## ⏱️ 예상 소요 시간

| 환경 | 에폭당 시간 | 총 시간 (30 epochs) |
|------|-------------|---------------------|
| GPU (NVIDIA RTX 3090) | ~15초 | ~7.5분 |
| GPU (NVIDIA V100) | ~20초 | ~10분 |
| CPU (16 cores) | ~2분 | ~60분 |

---

## 🎯 성능 목표

기존 Hybrid 모델과 비슷한 성능 목표:

| 지표 | 목표 | 설명 |
|------|------|------|
| **AUC** | > 0.85 | ROC AUC |
| **AP** | > 0.80 | Average Precision |
| **Accuracy** | > 0.80 | 이진 분류 정확도 |

---

## 🔧 하이퍼파라미터 튜닝 (선택사항)

더 좋은 성능을 원하면 `train_graphseal_fast.py`의 `Config` 클래스에서 조정:

```python
class Config:
    # 모델 크기 조정
    EMBEDDING_DIM = 256  # 더 크게: 512
    HIDDEN_DIM = 512     # 더 크게: 1024
    NUM_LAYERS = 4       # 더 깊게: 5 or 6
    
    # 학습 설정
    EPOCHS = 30          # 더 길게: 50
    BATCH_SIZE = 256     # GPU 메모리에 따라 조정
    LEARNING_RATE = 0.001  # 더 작게: 0.0005
```

---

## 🐛 트러블슈팅

### 문제 1: CUDA out of memory
```bash
# 해결: 배치 크기 줄이기
BATCH_SIZE = 128  # 또는 64
```

### 문제 2: 학습이 너무 느림 (CPU)
```bash
# 해결 1: Colab GPU 사용
# 해결 2: 에폭 수 줄이기
EPOCHS = 15  # 반으로 줄이기
```

### 문제 3: 입력 파일이 없음
```bash
# Phase 2를 먼저 실행해야 함
python phase2/main.py

# 또는 Phase 1+2를 모두 실행
python phase1/main_phase1.py
python phase2/main.py
```

---

## 📈 Phase 4로 이어지는 흐름

```
Phase 3 (GraphSEAL Fast)
    ↓
출력: node_embeddings_graphseal.pt
    ↓
Phase 4 (Rewiring) 입력으로 사용
    ↓
Buffer Capacity 계산 & Rewiring 최적화
```

**참고:** Phase 4는 현재 `node_embeddings_static.pt`를 사용하므로,  
GraphSEAL 출력을 사용하려면 Phase 4 코드에서 파일명만 변경하면 됩니다:

```python
# phase4/main_phase4.py의 Config 클래스
NODE_EMBEDDINGS = DATA_PROCESSED / "node_embeddings_graphseal.pt"  # 변경
```

---

## ✅ 체크리스트

- [ ] Phase 1-2가 완료되어 입력 파일들이 준비됨
- [ ] `train_graphseal_fast.py` 파일 생성 완료
- [ ] GPU 사용 가능 확인 (선택사항, CPU도 가능)
- [ ] 학습 실행: `python phase3/train_graphseal_fast.py`
- [ ] 출력 파일 확인: `phase3/output/graphseal_fast/`
- [ ] 로그에서 최종 AUC 확인 (> 0.85 목표)

---

**생성:** 2024-01-20  
**상태:** ✅ Ready to use  
**예상 시간:** ~10-60분 (GPU/CPU에 따라)
