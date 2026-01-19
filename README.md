# Supply Chain GNN Pipeline

공급망 링크 예측을 위한 3단계 GNN 파이프라인

## 📁 프로젝트 구조

```
GNN/
├── phase1/                    # Phase 1: 기업별 생산 레시피 추정
│   ├── src/                   # Phase 1 모듈
│   │   ├── b_matrix_generator.py
│   │   ├── inventory_module.py
│   │   ├── check_recipe.py
│   │   └── debug_deep_dive.py
│   ├── main_phase1.py         # 메인 실행 파일
│   ├── generate_dummy_data.py # 테스트 데이터 생성
│   ├── quick_test.sh          # 빠른 테스트
│   ├── requirements.txt       # 패키지 목록
│   ├── README.md              # 상세 문서
│   └── STRUCTURE.txt          # 구조 설명
│
├── phase2/                    # Phase 2: GraphSAGE 정적 임베딩
│   ├── src/                   # Phase 2 모듈
│   │   ├── graph_builder.py
│   │   ├── sampler.py
│   │   ├── GraphSAGE.py
│   │   ├── loss.py
│   │   └── trainer.py
│   ├── main_phase2.py         # 메인 실행 파일
│   ├── generate_phase2_dummy_data.py
│   ├── quick_test_phase2.sh
│   ├── test_phase2.py
│   ├── requirements.txt
│   ├── README.md
│   └── STRUCTURE.txt
│
├── phase3/                    # Phase 3: 하이브리드 링크 예측
│   ├── src/                   # Phase 3 모듈
│   │   ├── temporal_graph_builder.py
│   │   ├── sc_tgn.py
│   │   ├── graphseal.py
│   │   ├── hybrid_trainer.py
│   │   ├── loss.py
│   │   ├── negative_sampler.py
│   │   ├── benchmarks.py
│   │   ├── metrics.py
│   │   ├── robustness_test.py
│   │   ├── link_predictor.py
│   │   └── trainer_alt.py
│   ├── main.py                # 메인 실행 파일 (전체 학습)
│   ├── quick_test.py          # 빠른 테스트 (5 epochs)
│   ├── test.py                # 테스트 스크립트
│   ├── evaluate_comprehensive.py  # 종합 평가 (벤치마크 비교)
│   ├── generate_temporal_networks.py
│   ├── requirements.txt
│   ├── README.md
│   ├── STRUCTURE.txt
│   └── FINAL_SUMMARY.txt
│
├── data/                      # 데이터 디렉토리
│   ├── raw/                   # 원본 데이터
│   └── processed/             # 처리된 데이터
│
├── results/                   # 결과 디렉토리
│   ├── quick_test/            # Phase 3 빠른 테스트 결과
│   └── phase3/                # Phase 3 전체 학습 결과
│
└── README.md                  # 본 파일
```

## 🚀 Quick Start

### Phase 1: 생산 레시피 추정

```bash
cd phase1
bash quick_test.sh
```

**출력**: `../data/processed/disentangled_recipes.pkl` (기업별 33차원 레시피)

### Phase 2: GraphSAGE 임베딩

```bash
cd phase2
bash quick_test_phase2.sh
```

**출력**: 
- `../data/processed/node_embeddings_static.pt` (노드 임베딩)
- `../data/processed/train_edges.npy` (학습용 엣지)
- `../data/processed/test_edges.npy` (테스트용 엣지)

### Phase 3: 하이브리드 링크 예측

#### 빠른 테스트 (5 epochs)
```bash
cd phase3
python quick_test.py
```

#### 전체 학습 (100 epochs)
```bash
cd phase3
python main.py
```

#### 종합 평가 (벤치마크 비교)
```bash
cd phase3
python evaluate_comprehensive.py
```

## 📊 Phase별 설명

### Phase 1: Production Recipe Estimation
- **목적**: 기업별 생산 레시피(33차원) 추정
- **입력**: IO 테이블(A), 거래 네트워크(H), 기업 정보
- **출력**: 기업별 산업 중간재 사용 비율
- **방법**: Weighted/Simple/Bayesian 추정

### Phase 2: GraphSAGE Static Embeddings
- **목적**: 정적 그래프 임베딩 생성 (커리큘럼 학습)
- **입력**: Phase 1 레시피, 거래 네트워크
- **출력**: 노드 임베딩 (128차원), Train/Test 분할
- **방법**: GraphSAGE + Curriculum Learning (Random → Historical Negatives)

### Phase 3: Hybrid Link Prediction
- **목적**: 공급망 링크 예측 (두 트랙 하이브리드)
- **Track A**: SC-TGN (시계열 패턴)
- **Track B**: GraphSEAL (구조적 패턴)
- **Ensemble**: 두 트랙 결과 결합
- **손실**: TIS-aware BCE + Ranking Loss
- **평가**: Recall@K, MRR, RMSE, 벤치마크 비교 (PA/RA/JC)

## 📈 성능 평가

### Phase 3 종합 평가 지표

1. **Recall@K**: Top-K 후보 중 정답 비율
   - Recall@10, Recall@50, Recall@100

2. **MRR (Mean Reciprocal Rank)**: 정답의 평균 역순위
   - 높을수록 정답이 상위에 랭크

3. **RMSE**: 예측 확률과 실제 레이블 간 오차

4. **벤치마크 비교**:
   - Preferential Attachment (PA)
   - Resource Allocation (RA)
   - Jaccard Coefficient (JC)

5. **Robustness Test**: 네거티브 비율 변화에 따른 성능 분석

## 🔧 요구사항

### Phase 1
```bash
numpy, pandas, scipy, matplotlib
```

### Phase 2 & 3
```bash
torch, torch-geometric, pandas, numpy, scipy, matplotlib, tqdm
```

## 📝 실행 순서 (전체 파이프라인)

```bash
# 1. Phase 1 실행
cd phase1
python main_phase1.py

# 2. Phase 2 실행
cd ../phase2
python main_phase2.py

# 3. Phase 3 실행
cd ../phase3
python main.py

# 4. 종합 평가
python evaluate_comprehensive.py
```

## 🐛 트러블슈팅

### Phase 1
- **MemoryError**: `Config.BATCH_SIZE` 줄이기
- **산업코드 없음**: CSV 컬럼명 확인

### Phase 2
- **CUDA 메모리 부족**: `Config.BATCH_SIZE` 줄이기
- **Curriculum 단계 조정**: `Config.CURRICULUM_SCHEDULE` 수정

### Phase 3
- **학습 불안정**: Learning rate 낮추기
- **메모리 부족**: 작은 모델 사용 (quick_test.py 참고)

## 📖 상세 문서

각 Phase 디렉토리의 `README.md`를 참고하세요:

- `phase1/README.md`: Phase 1 상세 가이드
- `phase2/README.md`: Phase 2 상세 가이드
- `phase3/README.md`: Phase 3 상세 가이드

## 🎯 프로젝트 목표

1. **Data Leakage 방지**: Train/Val/Test 엄격 분리
2. **Realistic Negative Sampling**: Historical + Random 네거티브
3. **TIS-aware Learning**: 취약 기업 페널티
4. **Curriculum Learning**: 쉬운 샘플 → 어려운 샘플
5. **Hybrid Approach**: 시계열 + 구조적 정보 결합
6. **Comprehensive Evaluation**: 다양한 지표 + 벤치마크 비교

## 👥 기여

버그 리포트, 기능 제안은 이슈로 등록해주세요.

## 📄 라이선스

MIT License

---

**Made with ❤️ for Supply Chain Network Analysis**
