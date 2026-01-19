# Phase 4: Constrained Rewiring for Supply Chain Resilience

**목표:** TIS 기반 공급망 재배선 최적화

## 개요

Phase 4는 Phase 3에서 식별된 취약 노드들에 대해 제약 조건을 만족하는 새로운 공급 연결을 추가하여 공급망의 회복력을 향상시킵니다.

### 핵심 기능

1. **충격 완충력 계산**: TIS와 재무 데이터를 활용한 Buffer 계산
2. **제약 조건 검증**: Recipe 유사도, 용량 비율, Degree 제한 등
3. **최적화 알고리즘**: 다목적 최적화를 통한 최적 재배선
4. **벤치마크 비교**: Greedy, Random, TIS-Optimized 방법과 비교
5. **종합 평가**: 네트워크 메트릭 및 개선도 분석

## 디렉토리 구조

```
phase4/
├── main_phase4.py              # 메인 실행 스크립트
├── test_phase4.py              # 모듈 테스트 스크립트
├── requirements.txt            # Python 의존성
├── README.md                   # 설명서 (본 문서)
├── PHASE4_DESIGN.md            # 설계 문서
├── PHASE4_SUMMARY.md           # 요약 문서
├── config/
│   └── phase4_config.yaml      # 설정 파일
├── src/
│   ├── buffer_calculator.py    # 충격완충력 계산 모듈
│   ├── penalty_calculator.py   # Recipe/용량 패널티 계산
│   ├── constraint_checker.py   # 제약 조건 검증 모듈
│   ├── rewiring_optimizer.py   # 재배선 최적화 알고리즘
│   ├── benchmarks.py           # 벤치마크 방법들 (Greedy, Random, TIS-Optimized)
│   └── evaluate_rewiring.py    # 재배선 결과 평가 및 비교
├── output/                     # 출력 결과 저장 (자동 생성)
└── logs/                       # 로그 파일 (자동 생성)
```

## 설치

### 의존성 설치

```bash
cd phase4
pip install -r requirements.txt
```

### 필수 라이브러리

- numpy
- pandas
- torch (PyTorch)
- pyyaml
- scikit-learn

## 사용 방법

### 1. 기본 실행

```bash
python main_phase4.py --config config/phase4_config.yaml
```

### 2. 명령줄 옵션으로 실행

```bash
# 특정 연도, 방법, 파라미터 지정
python main_phase4.py --year 2024 --method optimization --top_k 100 --max_new_edges 50

# 다른 방법으로 실행
python main_phase4.py --method greedy --top_k 50
python main_phase4.py --method random --max_new_edges 30
python main_phase4.py --method tis_optimized --top_k 100
```

### 3. 모듈 테스트

```bash
# 전체 모듈 검증 (합성 데이터 사용)
python test_phase4.py
```

### 4. 설정 파일 수정

`config/phase4_config.yaml` 파일을 편집하여 다음 항목을 설정:

- **데이터 경로**: TG, 노드 특성, TIS 파일
- **재배선 파라미터**: 방법, top_k, max_new_edges, 가중치 (alpha, beta, gamma)
- **제약 조건**: degree 제한, recipe 유사도 임계값, 용량 비율
- **출력 설정**: 출력 디렉토리, 중간 결과 저장 여부
- **평가 설정**: 베이스라인 실행, Pareto 분석 등

## 입력 데이터

### 필수 파일

1. **Temporal Graph (TG)**: `data/processed/tg_{year}_filtered.csv`
   - Columns: `src` (또는 `source`, `supplier`), `dst` (또는 `destination`, `buyer`), ...

2. **Node Features**: `data/processed/posco_network_capital_consumergoods_removed_{year}.csv`
   - Columns: `node_id`, 재무 데이터 (자산, 매출, 자본 등), recipe 컬럼

3. **TIS Scores**: `phase3/output/tis_scores_{year}.csv`
   - Columns: `node_id`, `tis` (또는 `TIS`, `tis_score`)

### 데이터 컬럼 요구사항

- **재무 데이터**: `total_assets`, `revenue`, `equity`, `debt` 등 (유연하게 처리)
- **Recipe 데이터**: `recipe_*` 접두사를 가진 컬럼들 (산업/제품 분류)
- **TIS**: 노드별 취약도 점수 (높을수록 취약)

## 출력 결과

### 1. 새로운 엣지 파일

`output/new_edges_{method}_{year}_{timestamp}.csv`

- Columns: `supplier`, `buyer`
- 추가된 새로운 공급 연결

### 2. 엣지별 개선도

`output/edge_improvements_{method}_{year}_{timestamp}.csv`

- Columns: `supplier`, `buyer`, `improvement`, `buffer`, `penalty`
- 각 엣지의 기여도 분석

### 3. 요약 통계

`output/summary_{method}_{year}_{timestamp}.csv`

- 전체 결과 요약 (엣지 수, 개선도, 실행 시간 등)

### 4. 평가 보고서

`output/evaluation/`

- `report_{method}.txt`: 개별 방법 평가 보고서
- `method_comparison.csv`: 방법간 비교표
- `summary_report_{year}_{timestamp}.txt`: 종합 요약 보고서

### 5. 전체 결과 (선택)

`output/full_results_{method}_{year}_{timestamp}.pkl`

- 전체 결과 객체 (중간 결과, 히스토리 포함)

## 재배선 방법

### 1. Optimization (기본)

다목적 최적화를 통한 최적 재배선:
- **목적 함수**: α×Buffer - β×Recipe_Penalty - γ×Capacity_Penalty
- **제약 조건**: Degree, Recipe 유사도, 용량 비율, Self-loop 방지

### 2. Greedy

Greedy 방식으로 순차적 선택:
- 각 취약 노드에 대해 최고 점수의 공급자 선택
- 빠르지만 국소 최적해 가능성

### 3. Random

랜덤 선택 (베이스라인):
- 제약 조건만 만족하면 랜덤하게 연결
- 성능 하한선 제공

### 4. TIS-Optimized

TIS 감소에 최적화:
- TIS가 낮은 공급자와 Buffer에 집중
- 단순하지만 효과적

## 평가 메트릭

### 네트워크 회복력

- `weighted_avg_tis`: Degree로 가중된 평균 TIS (낮을수록 좋음)
- `high_risk_coverage`: 고위험 노드 커버리지 (높을수록 좋음)

### 연결성 메트릭

- `avg_degree`, `density`: 네트워크 밀도
- `degree_std`: Degree 분산도

### 효율성 메트릭

- `avg_tis_reduction_potential`: TIS 감소 잠재력
- `high_tis_buyer_ratio`: 고위험 구매자 비율

## 예제 워크플로우

```bash
# 1. 테스트 실행 (합성 데이터)
python test_phase4.py

# 2. 최적화 방법으로 실행
python main_phase4.py --method optimization --top_k 100 --max_new_edges 50

# 3. 모든 방법 비교 (config에서 run_baselines: true 설정)
python main_phase4.py --config config/phase4_config.yaml

# 4. 결과 확인
cat output/evaluation/summary_report_2024_*.txt
```

## 설계 문서

자세한 알고리즘 및 수식은 다음 문서 참조:

- **PHASE4_DESIGN.md**: 상세 설계 및 알고리즘 설명
- **PHASE4_SUMMARY.md**: 요약 및 모듈 개요

## 문제 해결

### 데이터 파일을 찾을 수 없음

- `config/phase4_config.yaml`에서 경로 확인
- `{year}` 플레이스홀더가 올바르게 치환되는지 확인

### 제약 조건이 너무 엄격함

- `config/phase4_config.yaml`의 `constraints` 섹션 완화:
  - `recipe_similarity_threshold` 낮추기 (0.5~0.7)
  - `capacity_ratio_min/max` 범위 확대
  - `max_supplier_outdegree/max_buyer_indegree` 증가

### 새로운 엣지가 너무 적음

- `max_new_edges` 증가
- `top_k` 증가 (더 많은 취약 노드 대상)
- 제약 조건 완화

### 메모리 부족

- `top_k` 감소
- `max_new_edges` 감소
- 노드 수가 많은 경우 샘플링 고려

## 향후 개선 방향

1. **고급 최적화**: Simulated Annealing, Genetic Algorithm
2. **동적 제약**: 시간에 따라 변하는 제약 조건
3. **다중 목표 Pareto 최적화**: 명시적 Pareto frontier 탐색
4. **시뮬레이션 검증**: 재배선 후 공급망 시뮬레이션
5. **시각화**: 네트워크 변화 및 개선도 시각화

## 라이선스 및 저자

- Author: Phase 4 Development Team
- Date: 2024

---

**Note**: 실제 데이터 실행 전 반드시 `test_phase4.py`로 모든 모듈을 검증하세요!
