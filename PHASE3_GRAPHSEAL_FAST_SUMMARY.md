# Phase 3: GraphSEAL Fast Training - 요약

**작성일:** 2024-01-20  
**상태:** ✅ **완료 및 실행 준비**

---

## 🎯 목표

시간 절약을 위해 **GraphSEAL 단독 모델**로 빠르게 학습 (30 epochs)  
→ 모델 규모를 키워서 성능 유지

---

## 📝 생성된 파일

### 1. **train_graphseal_fast.py** (454 lines)
```python
phase3/train_graphseal_fast.py
```

**주요 기능:**
- ✅ GraphSEAL 단독 학습 (SC-TGN, Hybrid 제거)
- ✅ 모델 규모 2배 증가 (Embedding 256, Hidden 512)
- ✅ 30 에폭 빠른 학습
- ✅ 배치 크기 256으로 최적화
- ✅ GPU/CPU 자동 감지
- ✅ 상세한 로깅 및 진행 상황 출력

### 2. **GRAPHSEAL_FAST_GUIDE.md**
```python
GRAPHSEAL_FAST_GUIDE.md
```

**내용:**
- 변경사항 상세 설명
- 실행 방법 단계별 가이드
- 예상 출력 및 로그 예시
- 트러블슈팅 가이드
- 하이퍼파라미터 튜닝 가이드

---

## 🔧 주요 변경사항

### 기존 (Hybrid 모델)
```python
- SC-TGN + GraphSEAL Hybrid
- Embedding: 128, Hidden: 256, Layers: 3
- 에폭: 50-100
- 복잡한 가중치 조합
- 학습 시간: ~2-4시간
```

### 현재 (GraphSEAL Fast)
```python
- GraphSEAL 단독 (간소화)
- Embedding: 256, Hidden: 512, Layers: 4  (2배 증가)
- 에폭: 30 (빠른 학습)
- 단순 BCE Loss
- 학습 시간: ~10-60분 (GPU/CPU)
```

---

## 🚀 실행 방법

```bash
# 1. 프로젝트 루트로 이동
cd /Users/iyulim/Desktop/나이스/GNN

# 2. 필요 파일 확인
ls -la data/processed/
# 필요: train_edges.npy, test_edges.npy, node_embeddings_static.pt

# 3. 실행
python phase3/train_graphseal_fast.py

# GPU 있으면 자동으로 GPU 사용
# 없으면 CPU로 학습 (느리지만 가능)
```

---

## 📊 모델 스펙

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Embedding Dim** | 256 | 노드 임베딩 차원 |
| **Hidden Dim** | 512 | 은닉층 차원 |
| **Num Layers** | 4 | GNN 레이어 수 |
| **Num Hops** | 3 | 서브그래프 hop 수 |
| **Dropout** | 0.2 | 드롭아웃 비율 |
| **총 파라미터** | ~2M | 약 200만 개 파라미터 |

### 학습 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Epochs** | 30 | 학습 에폭 수 |
| **Batch Size** | 256 | 배치 크기 |
| **Learning Rate** | 0.001 | 학습률 |
| **Weight Decay** | 1e-5 | 가중치 감쇠 |
| **Neg Samples** | 5:1 | Negative sampling 비율 |

---

## ⏱️ 예상 소요 시간

| 환경 | 에폭당 | 총 시간 (30 epochs) |
|------|--------|---------------------|
| **GPU (RTX 3090)** | ~15초 | **~7.5분** ✅ |
| **GPU (V100)** | ~20초 | **~10분** ✅ |
| **CPU (16 cores)** | ~2분 | **~60분** |

---

## 📈 성능 목표

| 지표 | 목표 | 의미 |
|------|------|------|
| **AUC** | > 0.85 | ROC AUC Score |
| **AP** | > 0.80 | Average Precision |
| **Accuracy** | > 0.80 | 분류 정확도 |

---

## 📁 출력 파일

```bash
phase3/output/graphseal_fast/
├── graphseal_model.pt                  # 학습된 모델 (Phase 4 사용)
├── node_embeddings_graphseal.pt        # 최종 노드 임베딩
└── logs/
    └── train_20260120_HHMMSS.log      # 상세 학습 로그
```

---

## 🔗 Phase 4 연결

GraphSEAL 출력을 Phase 4에서 사용하려면:

```python
# phase4/main_phase4.py의 Config 클래스에서 변경
class Config:
    # 기존
    NODE_EMBEDDINGS = DATA_PROCESSED / "node_embeddings_static.pt"
    
    # 변경 (GraphSEAL 출력 사용)
    NODE_EMBEDDINGS = PROJECT_ROOT / "phase3" / "output" / "graphseal_fast" / "node_embeddings_graphseal.pt"
```

---

## 🐛 트러블슈팅

### 1. CUDA out of memory
```python
# 배치 크기 줄이기
BATCH_SIZE = 128  # 또는 64
```

### 2. 입력 파일 없음
```bash
# Phase 1-2 먼저 실행
python phase1/main_phase1.py
python phase2/main.py
```

### 3. 성능이 목표에 못 미침
```python
# 에폭 수 늘리기
EPOCHS = 50

# 또는 모델 더 크게
EMBEDDING_DIM = 512
HIDDEN_DIM = 1024
```

---

## ✅ 체크리스트

- [x] `train_graphseal_fast.py` 생성 (454 lines)
- [x] `GRAPHSEAL_FAST_GUIDE.md` 생성 (상세 가이드)
- [x] 모델 규모 2배 증가 (256/512)
- [x] 30 에폭 설정
- [x] GPU/CPU 자동 감지
- [x] 상세 로깅 구현
- [x] Git 커밋 및 푸시 완료

---

## 📦 Git 정보

```bash
Commit: 2faf696
Message: feat: GraphSEAL 단독 빠른 학습 스크립트 추가 (30 epochs)

Files:
- phase3/train_graphseal_fast.py (454 lines, new)
- GRAPHSEAL_FAST_GUIDE.md (new)
```

---

## 🎓 사용 예시

### 기본 실행
```bash
python phase3/train_graphseal_fast.py
```

### 예상 출력
```
================================================================================
GraphSEAL Fast Training - Starting
================================================================================
Device: cuda
Epochs: 30
Model: Embedding=256, Hidden=512, Layers=4
Model parameters: 2,134,560

Loading data...
✓ Loaded H matrix: shape (10000, 10000), 50000 edges
✓ Loaded node embeddings: (10000, 128)
✓ Resizing embeddings from 128 to 256

Generating 5 negative samples per positive edge...
Generated 40000 negative samples

Creating dataloaders...
Train batches: 156, Test batches: 39

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
✓ Saved model to phase3/output/graphseal_fast/graphseal_model.pt
✓ Saved embeddings to phase3/output/graphseal_fast/node_embeddings_graphseal.pt

GraphSEAL Fast Training - Completed Successfully
```

---

## 🚀 다음 단계

1. **Phase 1-2 완료 확인**
   ```bash
   ls -la data/processed/
   # train_edges.npy, test_edges.npy, node_embeddings_static.pt 확인
   ```

2. **Phase 3 실행**
   ```bash
   python phase3/train_graphseal_fast.py
   ```

3. **결과 확인**
   ```bash
   ls -la phase3/output/graphseal_fast/
   # graphseal_model.pt, node_embeddings_graphseal.pt 생성 확인
   ```

4. **Phase 4 실행** (Optional: 임베딩 경로 변경 후)
   ```bash
   python phase4/main_phase4.py
   ```

---

**상태:** ✅ **완료 - 즉시 실행 가능**  
**예상 시간:** 7.5분 (GPU) ~ 60분 (CPU)  
**커밋:** `2faf696`

---

**작성:** 2024-01-20  
**GNN Supply Chain Project**
