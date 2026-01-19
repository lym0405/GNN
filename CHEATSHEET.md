# GNN 프로젝트 치트시트 (Cheat Sheet)

## 🚀 빠른 실행 가이드

### Phase 1: 레시피 추정
```bash
cd phase1
bash quick_test.sh                    # 빠른 테스트 (1,000개 기업)
python main_phase1.py                 # 전체 실행
```

### Phase 2: GraphSAGE 임베딩
```bash
cd phase2
bash quick_test_phase2.sh             # 빠른 테스트 (500개 기업, 10 epochs)
python main_phase2.py                 # 전체 실행
```

### Phase 3: 링크 예측
```bash
cd phase3
python quick_test.py                  # 빠른 테스트 (5 epochs)
python main.py                        # 전체 학습 (100 epochs)
python evaluate_comprehensive.py     # 종합 평가
```

## 📁 주요 경로

### 입력 데이터
```
../data/raw/A_33.csv                  # IO 테이블
../data/raw/H_csr_model2.npz          # 거래 네트워크
../data/raw/firm_to_idx_model2.csv    # 기업 인덱스
../data/raw/posco_network_20XX.csv    # 연도별 네트워크 (Phase 3)
```

### Phase 1 출력
```
../data/processed/disentangled_recipes.pkl    # 기업별 33차원 레시피
../data/processed/B_matrix.npy                # B 행렬
```

### Phase 2 출력
```
../data/processed/node_embeddings_static.pt   # 노드 임베딩
../data/processed/train_edges.npy             # 학습용 엣지
../data/processed/test_edges.npy              # 테스트용 엣지
```

### Phase 3 출력
```
../results/quick_test/hybrid_model_quick.pt   # 빠른 테스트 모델
../results/quick_test/phase3_metrics_quick.npz
../results/phase3/                            # 전체 학습 결과
```

## 🔧 주요 설정 파일

### Phase 1: phase1/main_phase1.py
```python
class Config:
    BATCH_SIZE = 10000              # 배치 크기
    METHOD = "weighted"             # 추정 방법: weighted, simple, bayesian
    ALPHA = 0.5                     # Bayesian alpha (if method='bayesian')
```

### Phase 2: phase2/main_phase2.py
```python
class Config:
    HIDDEN_DIM = 128                # 은닉층 차원
    OUTPUT_DIM = 128                # 출력 차원
    EPOCHS = 50                     # 총 에폭
    LR = 0.001                      # Learning rate
    CURRICULUM_SCHEDULE = {         # 커리큘럼 스케줄
        'random': (0, 10),
        'historical_easy': (10, 30),
        'historical_hard': (30, 50)
    }
```

### Phase 3: phase3/main.py
```python
class Config:
    EPOCHS = 100                    # 총 에폭
    LR = 0.0001                     # Learning rate
    BATCH_SIZE = 512                # 배치 크기
    
    # TGN 설정
    TGN_HIDDEN_DIM = 64
    TGN_TIME_DIM = 32
    
    # GraphSEAL 설정
    SEAL_HIDDEN_DIM = 64
    SEAL_NUM_HOPS = 2
    
    # Loss 설정
    TIS_ALPHA = 0.3                 # TIS 페널티 강도
    RANKING_WEIGHT = 0.1            # Ranking loss 가중치
```

## 📊 평가 지표

### Phase 3 평가
```python
# Recall@K
recall_10 = compute_recall_at_k(predictions, labels, k=10)
recall_50 = compute_recall_at_k(predictions, labels, k=50)
recall_100 = compute_recall_at_k(predictions, labels, k=100)

# MRR (Mean Reciprocal Rank)
mrr = compute_mrr(predictions, labels)

# RMSE
rmse = compute_rmse(predictions, labels)
```

## 🐛 트러블슈팅

### MemoryError (Phase 1)
```python
# main_phase1.py에서
Config.BATCH_SIZE = 5000  # 기본값 10000에서 줄이기
```

### CUDA Out of Memory (Phase 2, 3)
```python
# Config에서
BATCH_SIZE = 256          # 기본값 512에서 줄이기
# 또는 CPU 사용
device = 'cpu'
```

### ImportError
```bash
# 경로 문제인 경우
cd phase1  # 또는 phase2, phase3
python main_phase1.py  # Phase 폴더 내에서 실행
```

### 커리큘럼 학습 조정 (Phase 2)
```python
# 더 빠른 학습
CURRICULUM_SCHEDULE = {
    'random': (0, 5),
    'historical_easy': (5, 15),
    'historical_hard': (15, 25)
}
```

## 🧪 테스트 데이터 생성

### Phase 1 더미 데이터
```bash
cd phase1
python generate_dummy_data.py --n_firms 1000 --density 0.02
```

### Phase 2 더미 데이터
```bash
cd phase2
python generate_phase2_dummy_data.py --n_firms 500 --density 0.03
```

## 📈 결과 확인

### Phase 1 레시피 검증
```bash
cd phase1
python src/check_recipe.py ../data/processed/disentangled_recipes.pkl
```

### Phase 3 성능 시각화
```bash
cd phase3
python evaluate_comprehensive.py  # 자동으로 그래프 생성
# 출력: ../results/phase3/evaluation_results.png
```

## 🔍 디버깅

### Phase 1 개별 기업 분석
```bash
cd phase1
python src/debug_deep_dive.py ../data/processed/disentangled_recipes.pkl --firm <사업자번호>
```

### Phase 2 노드 임베딩 확인
```python
import torch
embeddings = torch.load('../data/processed/node_embeddings_static.pt')
print(embeddings.shape)  # [N, 128]
```

### Phase 3 모델 로드
```python
checkpoint = torch.load('../results/quick_test/hybrid_model_quick.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 📚 문서 참조

- **전체 가이드**: `/README.md`
- **Phase 1 상세**: `/phase1/README.md`
- **Phase 2 상세**: `/phase2/README.md`
- **Phase 3 상세**: `/phase3/README.md`
- **프로젝트 정리**: `/PROJECT_REORGANIZATION.txt`

## 💡 유용한 팁

### 1. 전체 파이프라인 한 번에 실행
```bash
cd phase1 && python main_phase1.py && \
cd ../phase2 && python main_phase2.py && \
cd ../phase3 && python main.py
```

### 2. 로그 저장
```bash
python main.py 2>&1 | tee training.log
```

### 3. 백그라운드 실행 (긴 학습)
```bash
nohup python main.py > training.log 2>&1 &
```

### 4. GPU 사용 확인
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 5. 메모리 사용량 모니터링
```bash
# GPU
watch -n 1 nvidia-smi

# CPU/RAM
htop
```

## 🎯 체크리스트

### Phase 1 실행 전
- [ ] `../data/raw/`에 A_33.csv 있음
- [ ] `../data/raw/`에 H_csr_model2.npz 있음
- [ ] `../data/raw/`에 기업 정보 CSV 있음

### Phase 2 실행 전
- [ ] Phase 1 완료 (disentangled_recipes.pkl 존재)
- [ ] `../data/raw/`에 H_csr_model2.npz 있음
- [ ] GPU/CPU 선택 완료

### Phase 3 실행 전
- [ ] Phase 1, 2 완료
- [ ] `../data/raw/`에 posco_network_20XX.csv 있음 (2020-2023)
- [ ] node_embeddings_static.pt 존재
- [ ] GPU/CPU 선택 완료

---

**마지막 업데이트**: 2026년 1월 19일
