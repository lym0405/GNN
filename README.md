# GNN 공급망 네트워크 분석 프로젝트

**Graph Neural Networks for Supply Chain Resilience Analysis**

공급망 네트워크의 취약성 분석 및 재배선 최적화를 위한 GNN 기반 파이프라인

---

## 📋 프로젝트 개요

### 목표
1. **공급망 리스크 예측**: 특정 기업의 충격이 전체 네트워크에 미치는 영향 분석
2. **재배선 경로 추천**: 충격 발생 시 최적의 대체 거래처 제안
3. **Historical Validation**: 실제 역사적 사건(2019 일본 수출규제)으로 모델 검증

### 방법론
- **Phase 1**: 생산함수 추정 (IO Table 기반 B-Matrix)
- **Phase 2**: 정적 그래프 임베딩 (Static GNN)
- **Phase 3**: 동적 링크 예측 (Temporal GNN + GraphSEAL)
- **Phase 4**: 제약 조건 고려 재배선 (Constrained Rewiring)
- **Phase 5**: 역사적 검증 (2019 Japan Export Restriction)

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
```
data/
├── raw/
│   ├── posco_network_capital_consumergoods_removed_2024.csv  # 거래 네트워크
│   ├── vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv  # 기업 정보
│   ├── final_tg_2024_estimation.csv  # 매출 데이터
│   ├── export_estimation_value_final.csv  # 수출액
│   ├── asset_final_2024_6차.csv  # 자산
│   ├── shock_after_P_v2.csv  # TIS 리스크
│   ├── A_33.csv  # IO 계수 행렬
│   └── firm_to_idx_model2.csv  # 기업 ID 매핑
└── processed/  # 중간 결과물 저장
```

### 3. 전체 파이프라인 실행
```bash
# Phase 1: 생산함수 추정
cd phase1
python main_phase1.py

# Phase 2: 정적 임베딩
cd ../phase2
python main_phase2.py

# Phase 3: 링크 예측 (학습 + 평가)
cd ../phase3
python main.py

# Phase 4: 제약 조건 재배선
cd ../phase4
python main_phase4.py

# Phase 5: 역사적 검증 (옵션)
cd ../phase5
python main_phase5.py
```

---

## 📊 파이프라인 상세

### Phase 1: 생산함수 추정 (B-Matrix Generator)
**목적**: IO Table 기반으로 각 기업의 투입 구조(레시피) 추정

**입력**:
- 기업 정보 (`IO상품_단일_대분류_코드`)
- 매출 데이터 (`tg_2024_final`)
- IO 계수 행렬 (`A_33.csv`, 33×33)

**출력**:
- `B_matrix.npy`: 기업별 투입 계수 행렬 (N×33)
- `firm_to_idx_model2.csv`: 기업 ID → 인덱스 매핑
- `H_matrix.npy`: 기업-산업 매핑 행렬 (N×33)

**실행**:
```bash
cd phase1
python main_phase1.py
```

**주요 기능**:
- IO 테이블 33개 산업 분류 기반
- RAS 알고리즘으로 기업별 레시피 추정
- 매출 기반 Share 계산

---

### Phase 2: 정적 그래프 임베딩 (Static GNN)
**목적**: 네트워크 구조와 기업 특성을 반영한 임베딩 생성

**입력**:
- 거래 네트워크 (`posco_network_capital_consumergoods_removed_2024.csv`)
- Phase 1 출력 (B-Matrix)
- 기업 features (좌표, 매출, 수출, 자산, TIS)

**출력**:
- `node_embeddings_static.pt`: 노드 임베딩 (N×128)
- `train_edges.npy`, `test_edges.npy`: Train/Test 분할
- `tis_score_normalized.npy`: 정규화된 TIS 점수

**실행**:
```bash
cd phase2
python main_phase2.py
```

**모델**:
- **GCN** (Graph Convolutional Network)
- Input: 노드 features (좌표, 매출, B-Matrix 등)
- Hidden: 256 → 128 dim
- 학습: Link Prediction (엣지 존재 여부)

**최적화**:
- Negative sampling ratio: 1:2 (Positive:Negative)
- Batch size: 4096
- Epochs: 100 (early stopping)

---

### Phase 3: 동적 링크 예측 (Temporal GNN)
**목적**: 시간에 따른 거래 관계 변화 예측 및 재배선 경로 추천

**입력**:
- Phase 2 출력 (Static embeddings)
- 시계열 네트워크 (2020~2024)
- Historical negative samples (과거 거래 데이터)

**출력**:
- `hybrid_model_best.pt`: 학습된 모델
- `phase3_metrics.npz`: 평가 지표
- 재배선 추천 리스트

**실행**:
```bash
cd phase3
python main.py
```

**모델**: Two-Track Hybrid Architecture
1. **Track A (SC-TGN)**: Temporal Graph Network
   - Memory module로 시간 정보 반영
   - Message passing으로 이웃 정보 집계
   
2. **Track B (GraphSEAL)**: Structural + UKGE
   - Subgraph 기반 링크 예측
   - Uncertainty-aware 신뢰도 추정

3. **Ensemble**: α·Track_A + (1-α)·Track_B

**Negative Sampling**:
- **Historical (50%)**: 2020-2023년 과거 거래 데이터
- **Random (50%)**: 무작위 샘플링
- 총 14,550개 historical negatives 사용

**평가**:
- AUC, AP (Average Precision)
- Recall@K (K=10, 50, 100, 500, 1000)

---

### Phase 4: 제약 조건 재배선 (Constrained Rewiring)
**목적**: 실질적 제약(버퍼, 재고, 물류)을 고려한 재배선

**입력**:
- Phase 3 출력 (링크 예측 점수)
- 기업 버퍼/재고 정보
- 물류/규제 제약

**출력**:
- 최적화된 재배선 계획
- 제약 위반 체크 리포트
- 비용/효익 분석

**실행**:
```bash
cd phase4
python main_phase4.py
```

**주요 모듈**:
1. `buffer_calculator.py`: 버퍼 계산
2. `penalty_calculator.py`: 위험도 페널티
3. `constraint_checker.py`: 제약 조건 검증
4. `rewiring_optimizer.py`: 최적화 솔버
5. `evaluate_rewiring.py`: 성능 평가

**제약 조건**:
- 버퍼 용량 (생산 능력)
- 재고 수준
- 물류 거리/비용
- 규제/정책 제한

---

### Phase 5: 역사적 검증 (Historical Back-testing)
**목적**: 2019년 일본 수출규제 시뮬레이션으로 모델 검증

**시나리오**: 2019년 7월 일본의 반도체 소재 수출 규제
- **대상 소재**: 불화수소, 포토레지스트, 불화폴리이미드
- **영향 기업**: 삼성전자, SK하이닉스 등 반도체 기업

**입력**:
- 2018년 네트워크 (충격 이전)
- 2020년 네트워크 (재배선 결과)
- KSIC 기반 충격 대상 기업 선정

**출력**:
- Hit Rate@K, Recall@K, Precision@K
- 매출 감소 완화도
- 결과 보고서

**실행**:
```bash
cd phase5
python main_phase5.py
```

**평가 지표**:
- **예측력**: 실제 재배선 경로를 맞췄는가?
- **안정성**: 모델 추천이 실제보다 더 나은 결과를 가져왔는가?

---

## 📁 프로젝트 구조

```
GNN/
├── data/
│   ├── raw/                 # 원본 데이터
│   └── processed/           # 중간 결과물
├── results/                 # 최종 출력
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   └── phase5/
├── phase1/                  # 생산함수 추정
│   ├── main_phase1.py
│   └── src/
│       └── b_matrix_generator.py
├── phase2/                  # 정적 임베딩
│   ├── main_phase2.py
│   └── src/
│       ├── graph_builder.py
│       ├── feature_engineer.py
│       ├── model.py
│       └── trainer.py
├── phase3/                  # 동적 링크 예측
│   ├── main.py
│   └── src/
│       ├── temporal_graph_builder.py
│       ├── sc_tgn.py
│       ├── graphseal.py
│       ├── hybrid_trainer.py
│       └── negative_sampler.py
├── phase4/                  # 제약 조건 재배선
│   ├── main_phase4.py
│   └── src/
│       ├── buffer_calculator.py
│       ├── penalty_calculator.py
│       ├── constraint_checker.py
│       ├── rewiring_optimizer.py
│       └── evaluate_rewiring.py
├── phase5/                  # 역사적 검증
│   ├── main_phase5.py
│   └── src/
│       ├── ksic_matcher.py
│       ├── shock_injector.py
│       └── evaluator.py
├── README.md                # 이 파일
├── PROJECT_STATUS.md        # 진행 상황
├── PROJECT_STRUCTURE_SUMMARY.md  # 구조 요약
└── requirements.txt         # 의존성
```

---

## 🔧 주요 설정

### Phase 1 설정
```python
# phase1/main_phase1.py
IO_SECTORS = 33              # IO 테이블 산업 수
RAS_MAX_ITER = 100          # RAS 최대 반복
RAS_TOL = 1e-6              # 수렴 허용오차
```

### Phase 2 설정
```python
# phase2/main_phase2.py
HIDDEN_DIM = 256            # GCN hidden 차원
EMBEDDING_DIM = 128         # 최종 임베딩 차원
NEG_RATIO = 2               # Negative sampling ratio (1:2)
BATCH_SIZE = 4096           # 배치 크기
EPOCHS = 100                # 최대 에폭
LEARNING_RATE = 0.001       # 학습률
```

### Phase 3 설정
```python
# phase3/main.py
TGN_MEMORY_DIM = 128        # TGN memory 차원
GRAPHSEAL_HIDDEN_DIM = 128  # GraphSEAL hidden 차원
ENSEMBLE_ALPHA = 0.5        # Track A 가중치
HISTORICAL_RATIO = 0.5      # Historical negatives 비율
NEG_RATIO = 1.0             # Negative per positive
BATCH_SIZE = 1024           # 배치 크기
EPOCHS = 100                # 최대 에폭
```

---

## 📈 성능 지표

### Phase 2 (Static GNN)
- **Train AUC**: >0.95
- **Test AUC**: >0.85
- **학습 시간**: ~10-20분 (4096 batch)

### Phase 3 (Temporal GNN)
- **Train AUC**: >0.90
- **Test AUC**: >0.80
- **Recall@100**: >0.30
- **Historical Negatives**: 14,550개 (2020-2023)

### Phase 5 (Historical Validation)
- **목표 Hit Rate@100**: >30%
- **목표 Recall@1000**: >50%
- **매출 감소 완화**: >20%

---

## 🐛 문제 해결

### 1. Historical Negatives가 0개로 나오는 경우
**원인**: `firm_to_idx_model2.csv`의 컬럼명 불일치

**해결**:
```bash
# firm_to_idx_model2.csv 확인
head data/raw/firm_to_idx_model2.csv

# 컬럼명이 '사업자등록번호', 'idx'인지 확인
# 아니면 phase3/src/negative_sampler.py의 컬럼명 매칭 로직 수정
```

### 2. Phase 2 학습이 너무 느린 경우
**해결**: Batch size를 줄이거나 negative ratio 조정
```python
# phase2/main_phase2.py
BATCH_SIZE = 2048  # 4096 → 2048로 감소
NEG_RATIO = 1      # 2 → 1로 감소
```

### 3. 메모리 부족 (OOM)
**해결**: 데이터 크기 또는 모델 크기 축소
```python
# phase2/main_phase2.py
HIDDEN_DIM = 128      # 256 → 128
EMBEDDING_DIM = 64    # 128 → 64
```

### 4. 캐시 관련 문제
**해결**: 캐시 초기화
```bash
python clear_cache.py
```

---

## 📚 참고 자료

### 논문/문헌
1. Inoue, H., & Todo, Y. (2019). "Firm-level propagation of shocks through supply-chain networks"
2. Barrot, J. N., & Sauvagnat, J. (2016). "Input specificity and the propagation of idiosyncratic shocks"
3. 오석진, 이창민 (2020). "수출규제 3품목 수입동향 분석"
4. 한국은행 (2019). "일본의 수출규제 조치의 영향과 시사점"

### GNN 모델
- **GCN**: Kipf & Welling (2017)
- **TGN**: Rossi et al. (2020)
- **GraphSAINT**: Zeng et al. (2020)

### 관련 문서
- `PROJECT_STATUS.md`: 현재 진행 상황
- `PROJECT_STRUCTURE_SUMMARY.md`: 상세 구조
- `COLUMN_NAME_UPDATE.md`: 데이터 컬럼명 정리
- `PHASE3_HISTORICAL_NEGATIVES_FIX.md`: Phase 3 버그 수정 내역
- `CACHE_GUIDE.md`: 캐시 시스템 가이드

---

## 🤝 기여

이 프로젝트는 공급망 리스크 분석을 위한 연구 프로젝트입니다.

---

## 📝 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다.

---

## 📞 문의

문제나 질문이 있으시면 Issue를 등록해주세요.

---

**마지막 업데이트**: 2026년 1월 19일  
**버전**: 1.0  
**상태**: Phase 1-5 완료, 테스트 및 검증 진행 중
