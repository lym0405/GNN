# Phase 2: Static Graph Embedding

커리큘럼 학습 기반 GraphSAGE 임베딩 생성

## 📋 개요

Phase 1의 레시피 + 재무/TIS 데이터를 결합하여 GraphSAGE로 32차원 노드 임베딩 학습

- **입력**: Phase 1 레시피, 매출/수출/자산, TIS, H 행렬
- **출력**: 32차원 임베딩, Train/Test 엣지
- **특징**: 커리큘럼 학습 (Easy → Hard)

## 🚀 빠른 시작

### 1. Phase 1 완료 확인

```bash
ls data/processed/disentangled_recipes.pkl
```

### 2. 패키지 설치

```bash
pip install -r requirements_phase2.txt
```

### 3. Phase 2 실행

```bash
python main_phase2_fixed.py
```

### 4. 출력 확인

```
data/processed/
├── node_embeddings_static.pt    # 32차원 임베딩
├── train_edges.npy               # 학습 엣지 (80%)
├── test_edges.npy                # 평가 엣지 (20%)
└── X_feature_matrix.npy          # 피처 행렬
```

## 📊 피처 구조

### 간소화 버전 (73차원) - 기본값

```
┌─────────────────────┬────────┬──────────────────────┐
│ 피처 그룹           │ 차원   │ 설명                 │
├─────────────────────┼────────┼──────────────────────┤
│ 재무                │ 4      │ 매출/수출/자산/비율  │
│ 지리                │ 2      │ 위도/경도            │
│ 리스크              │ 1      │ TIS 점수             │
│ 산업                │ 33     │ One-Hot              │
│ 레시피              │ 33     │ Phase 1 출력         │
├─────────────────────┼────────┼──────────────────────┤
│ 총합                │ 73     │                      │
└─────────────────────┴────────┴──────────────────────┘
```

## 🎯 커리큘럼 학습 전략

### Easy Phase (Epoch 1-20)
- Random Negative: 100%
- Historical Hard Negative: 0%
- **목적**: 모델 안정화, 기본 패턴 학습

### Medium Phase (Epoch 21-50)
- Random Negative: 80%
- Historical Hard Negative: 20%
- **목적**: 점진적 난이도 상승, Hard 케이스 도입

### Hard Phase (Epoch 51-55)
- Random Negative: 60%
- Historical Hard Negative: 40%
- **목적**: 어려운 케이스 집중 학습

### Final Phase (Epoch 56-60)
- Random Negative: 70%
- Historical Hard Negative: 30%
- **목적**: 안정화 및 일반화 성능 향상

## ⚙️ 하이퍼파라미터

### 모델

```python
HIDDEN_DIM = 64        # 은닉층 차원
OUTPUT_DIM = 32        # 출력 임베딩 차원
DROPOUT = 0.3          # 드롭아웃
```

### 학습

```python
EPOCHS = 60            # 전체 에폭
BATCH_SIZE = 1024      # 배치 크기
LEARNING_RATE = 0.001  # 학습률
WEIGHT_DECAY = 1e-5    # L2 정규화
```

### 데이터

```python
TRAIN_RATIO = 0.8      # Train/Test 비율
RANDOM_SEED = 42       # 재현성
TIS_ALPHA = 0.3        # TIS 페널티 강도
```

## 🔧 설정 변경

`main_phase2_fixed.py`의 `Config` 클래스 수정:

```python
class Config:
    # 간소화 vs 전체 피처
    USE_SIMPLE_FEATURES = True  # False: 197차원
    
    # 커리큘럼 에폭 조정
    EASY_EPOCHS = 5
    MEDIUM_EPOCHS = 5
    HARD_EPOCHS = 5
    FINAL_EPOCHS = 5
    
    # TIS 사용 여부
    TIS_ALPHA = 0.3  # 0: TIS 무시
    
    # GPU 사용
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 📈 예상 성능

### 더미 데이터 (1,000 기업)
- 피처 생성: ~1초
- GraphSAGE 학습: ~90초 (60 epochs, CPU)
- 총 실행 시간: ~100초

### 실제 데이터 (438,946 기업)
- 피처 생성: ~2분
- GraphSAGE 학습: ~90분 (60 epochs, GPU)
- 총 실행 시간: ~95분

## 🔍 Data Leakage 방지

✅ **Train/Test 완전 분리**
- 2024년 엣지를 80/20으로 분할
- Test 엣지는 학습에 미사용

✅ **Message Passing은 Train만**
- GraphSAGE는 Train 엣지로만 학습
- Test 엣지는 평가 시에만 등장

✅ **Historical은 2020-2023만**
- 2024년 정보 완전 차단

## 🐛 문제 해결

### "CUDA out of memory"
→ `BATCH_SIZE`를 줄이거나 (512, 256), CPU로 실행

### "Phase 1 출력이 없습니다"
→ Phase 1을 먼저 실행: `python main_phase1.py`

### "Historical Negatives가 없습니다"
→ 2020-2023 데이터가 없으면 Random만 사용 (정상 작동)

### "학습이 너무 느립니다"
→ GPU 사용 권장, `BATCH_SIZE` 증가

## 📊 모니터링

학습 중 출력 예시:

```
Epoch 01/60 | Loss: 0.6234 | Pos: 0.612 | Neg: 0.387 | Val Acc: 0.589
Epoch 20/60 | Loss: 0.5523 | Pos: 0.681 | Neg: 0.318 | Val Acc: 0.681
Epoch 21/60 | Loss: 0.5412 | Pos: 0.695 | Neg: 0.305 | Val Acc: 0.695  # Medium 시작
Epoch 50/60 | Loss: 0.4567 | Pos: 0.756 | Neg: 0.234 | Val Acc: 0.761
Epoch 51/60 | Loss: 0.4456 | Pos: 0.763 | Neg: 0.227 | Val Acc: 0.768  # Hard 시작
Epoch 55/60 | Loss: 0.4223 | Pos: 0.778 | Neg: 0.215 | Val Acc: 0.781
Epoch 60/60 | Loss: 0.4012 | Pos: 0.781 | Neg: 0.219 | Val Acc: 0.781  # Final
```

**좋은 학습 신호:**
- Loss 감소
- Pos Score 증가 (→ 1.0)
- Neg Score 감소 (→ 0.0)
- Val Acc 증가

## 📚 다음 단계

Phase 2 완료 후:

1. **임베딩 확인**: `node_embeddings_static.pt`
2. **Phase 3 실행**: `python main_phase3_train.py`
3. **벤치마크**: `python main_phase3_benchmark.py`

## 💡 팁

### GPU 가속
```bash
# GPU 사용 가능 확인
python -c "import torch; print(torch.cuda.is_available())"

# PyTorch GPU 버전 설치 (예: CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 더미 데이터로 빠른 테스트
```bash
# Phase 1 더미 데이터 생성 (1,000 기업)
python generate_dummy_data.py --n_firms 1000

# Phase 1 실행
python main_phase1.py

# Phase 2 실행
python main_phase2_fixed.py
```

## 📖 참고

- **GraphSAGE**: Hamilton et al., "Inductive Representation Learning on Large Graphs"
- **Curriculum Learning**: Bengio et al., "Curriculum Learning"
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

---

**기여**: 버그 제보 및 개선 제안 환영합니다!
