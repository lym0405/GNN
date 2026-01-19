# Phase 5 Design Document: Historical Back-testing

## 1. 개요

### 목적
과거 실제 공급망 충격 사건을 시뮬레이션하여 GNN 모델의 예측력과 재배선 추천 성능을 검증

### Case Study
**2019년 7월 일본 수출규제**: 반도체 핵심 소재 3종 (불화수소, 포토레지스트, 불화폴리이미드)

### 검증 질문
1. **예측력**: 모델이 실제로 재배선된 거래처를 예측할 수 있는가?
2. **안정성**: 모델 추천대로 재배선했다면 실제보다 더 나은 결과를 얻었을까?

---

## 2. 방법론

### 2.1 시간축 설정

```
2018년          2019년 7월         2020년
  ↓               ↓                 ↓
[Before]  →  [Shock Event]  →  [After]
  ↑               ↑                 ↑
학습 데이터      충격 주입        검증 데이터
```

- **2018년**: 충격 이전 네트워크 (모델 입력)
- **2019년 7월**: 충격 사건 (시뮬레이션)
- **2020년**: 재배선 결과 (Ground Truth)

### 2.2 파이프라인

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: KSIC 매칭                                        │
│ - 공급자 기업 선정 (C20129, C20119, C20499, ...)        │
│ - 수요자 기업 선정 (C26110, C26120, ...)                │
│ - 알려진 기업 (솔브레인, SK하이닉스, ...)                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: 충격 주입 (Shock Injection)                      │
│ - Method A: 엣지 삭제 (일본→한국 거래 끊김)              │
│ - Method B: 노드 장애 (공급자 매출 = 0)                  │
│ - Method C: 복합 (A + B)                                 │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: 재배선 예측 (Phase 3 모델)                       │
│ - Input: 2018년 그래프 + 충격 시나리오                   │
│ - Output: 추천 대체 거래처 (Top-K)                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: 평가 (2020년 실제 데이터와 비교)                  │
│ - Hit Rate@K, Recall@K, Precision@K                      │
│ - Resilience Score (매출 감소 완화도)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 데이터 요구사항

### 3.1 필수 데이터

| 파일 | 설명 | 용도 |
|-----|------|-----|
| `posco_network_2018.csv` | 2018년 거래 네트워크 | 충격 이전 베이스라인 |
| `posco_network_2020.csv` | 2020년 거래 네트워크 | 재배선 결과 (Ground Truth) |
| `firm_to_idx_model2.csv` | 기업 ID → 인덱스 매핑 | 그래프 노드 변환 |

### 3.2 선택 데이터

| 파일 | 설명 | 용도 |
|-----|------|-----|
| `firm_info.csv` | 기업 정보 (KSIC 코드 포함) | KSIC 기반 기업 선정 |
| `posco_network_2019.csv` | 2019년 거래 네트워크 | 충격 직후 분석 (Optional) |
| `stock_prices.csv` | 주가 데이터 | 이벤트 스터디 (Optional) |

### 3.3 KSIC 코드

**공급자 (소재 생산)**
- `C20129`: 기타 기초 무기화학 물질 제조업
- `C20119`: 기타 기초 유기 화학물질 제조업
- `C20499`: 그 외 기타 분류 안된 화학제품 제조업
- `C20122`: 산소, 질소 및 기타 산업용 가스 제조업
- `C20501`: 합성섬유 제조업

**수요자 (반도체/디스플레이)**
- `C26110`: 전자집적회로 제조업
  - `C26111`: 메모리용
  - `C26112`: 비메모리용
- `C26120`: 다이오드, 트랜지스터 및 유사 반도체소자 제조업
  - `C26121`: 발광 다이오드
  - `C26129`: 기타 반도체 소자

**필수 기업 (Known Firms)**

공급자:
- 솔브레인 (LI5265)
- 램테크놀러지 (240623)
- 이엔에프테크놀로지 (215316)
- 동진쎄미켐 (355950)
- 코오롱인더스트리 (F03302)
- SKC (350885)
- SK머티리얼즈 (093025)

수요자:
- 삼성전자 (380725)
- SK하이닉스 (383511)
- MEMC코리아 (452556)
- 서울반도체 (092819)
- SK실트론 (360651)

---

## 4. 충격 시나리오

### 4.1 Option A: 엣지 삭제

**방법**: 공급자 → 수요자 간 거래 엣지 삭제

```python
# Pseudocode
shocked_graph = original_graph.copy()
edges_to_delete = [
    (supplier, buyer) 
    for supplier in supplier_nodes
    for buyer in buyer_nodes
    if (supplier, buyer) in original_graph.edges
]
shocked_graph.remove_edges(edges_to_delete)
```

**장점**: 단순, 명확, 해석 용이  
**단점**: 2차 효과 (공급자의 다른 고객 영향) 미반영

### 4.2 Option B: 노드 장애

**방법**: 공급자의 매출/생산 능력을 0으로 설정

```python
# Pseudocode
shocked_features = original_features.copy()
for supplier in supplier_nodes:
    shocked_features[supplier]['total_sales'] = 0
    shocked_features[supplier]['production_capacity'] = 0
```

**장점**: 2차 효과 반영, 현실적  
**단점**: Feature 의존적, 복잡

### 4.3 Option C: 복합

**방법**: A + B

**장점**: 가장 현실적  
**단점**: 복잡, 해석 어려움

**권장**: 초기에는 A (엣지 삭제)로 시작, 이후 C로 확장

---

## 5. 평가 지표

### 5.1 예측력 (Prediction Accuracy)

#### Hit Rate@K
```
Hit Rate@K = |Top-K ∩ Actual New Edges| / min(K, |Actual New Edges|)
```

Top-K 예측 중에서 실제 신규 엣지가 몇 개 포함되었는가?

**목표**: Hit Rate@100 > 0.3 (30% 이상)

#### Recall@K
```
Recall@K = |Top-K ∩ Actual New Edges| / |Actual New Edges|
```

실제 신규 엣지 중에서 Top-K에 포함된 비율

**목표**: Recall@1000 > 0.5 (50% 이상)

#### Precision@K
```
Precision@K = |Top-K ∩ Actual New Edges| / K
```

Top-K 예측 중에서 실제로 맞춘 비율

**목표**: Precision@100 > 0.1 (10% 이상)

### 5.2 안정성 (Resilience)

#### 매출 감소 완화도
```
Improvement = (Actual Reduction - Predicted Reduction) / Actual Reduction
```

모델 추천대로 재배선했을 때 매출 감소가 얼마나 완화되는가?

**목표**: Improvement > 0.2 (20% 이상 완화)

#### 네트워크 연결성
```
Connectivity = |Connected Components|
```

재배선 후 네트워크가 얼마나 연결되어 있는가?

**목표**: 연결 컴포넌트 개수가 실제보다 적음

---

## 6. 구현 상세

### 6.1 모듈 구조

```
phase5/
├── src/
│   ├── ksic_matcher.py          # KSIC 기반 기업 매칭
│   ├── shock_injector.py        # 충격 시나리오 생성
│   ├── predictor.py             # 재배선 예측 (Phase 3 모델 사용)
│   ├── evaluator.py             # 성능 평가
│   └── visualizer.py            # 결과 시각화
├── main_phase5.py               # 메인 스크립트
└── README.md                    # 사용 설명서
```

### 6.2 주요 클래스

#### KSICMatcher
```python
class KSICMatcher:
    """KSIC 코드로 기업 선정"""
    def get_firms_by_ksic(ksic_codes, exact_match=False)
    def get_firm_indices_by_ksic(ksic_codes)
    def get_known_firms_indices(firm_ids)
```

#### ShockInjector
```python
class ShockInjector:
    """충격 시나리오 생성"""
    def inject_edge_deletion(source_indices, target_indices)
    def inject_node_disruption(node_indices)
    def inject_supply_cut(supplier_indices, buyer_indices)
```

#### Phase5Evaluator
```python
class Phase5Evaluator:
    """성능 평가"""
    def compute_hit_rate_at_k(k_list)
    def compute_recall_at_k(k_list)
    def compute_precision_at_k(k_list)
    def compute_all_metrics()
```

---

## 7. 실행 계획

### Phase 1: 기본 구현 (현재)
- [x] KSIC 매칭 모듈
- [x] 충격 주입 모듈
- [x] 평가 모듈
- [x] 메인 스크립트
- [x] 문서화

### Phase 2: 데이터 준비
- [ ] firm_info.csv 생성 (KSIC 코드 추가)
- [ ] 2018, 2020년 네트워크 데이터 확인
- [ ] 필수 기업 사업자등록번호 확인

### Phase 3: 모델 통합
- [ ] Phase 3 모델 로드
- [ ] 충격 시나리오 입력 변환
- [ ] 재배선 예측 실행

### Phase 4: 평가 및 분석
- [ ] 실제 데이터와 비교
- [ ] 메트릭 계산
- [ ] 결과 시각화

### Phase 5: 보고서 작성
- [ ] 정량적 결과 정리
- [ ] 정성적 분석 (어떤 기업이 영향 받았나?)
- [ ] 제한사항 및 개선 방향

---

## 8. 기대 결과

### 정량적
- **Hit Rate@100**: 30-40% (100개 추천 중 30-40개 적중)
- **Recall@1000**: 50-60% (실제 신규 거래의 50-60% 포착)
- **Precision@100**: 10-15% (100개 중 10-15개 정확)
- **매출 완화**: 20-30% (모델 추천 시 감소폭 완화)

### 정성적
- 불화수소 공급자 (솔브레인 등)의 거래 변화 패턴
- 국산화 vs. 수입 대체 경로 분석
- 네트워크 구조 변화 (중심성, 클러스터링)

---

## 9. 제한사항

### 데이터
- 2018-2020년 데이터가 모두 필요
- KSIC 코드 정확도에 의존
- 거래 금액 정보 필요 (매출 분석용)

### 모델
- 정책, 정치적 결정은 반영 불가
- 단기 충격만 시뮬레이션 (장기 효과 미반영)
- 외부 요인 (환율, 다른 경제 충격) 분리 어려움

### 평가
- 인과관계 vs. 상관관계 구분 어려움
- Counterfactual (만약 ~했다면) 검증 불가능
- 소규모 사건 (특정 기업 파산 등)에는 부적합

---

## 10. 확장 가능성

### 다른 시나리오
1. **2021년 요소수 사태**: 중국 → 한국 요소수 공급 차단
2. **2020년 코로나19**: 글로벌 공급망 마비
3. **특정 기업 파산**: 1차 공급업체 파산 시뮬레이션

### 추가 분석
1. **업종별 취약성**: 어떤 산업이 가장 취약한가?
2. **지역별 영향**: 수도권 vs. 지방 영향 차이
3. **시간 동학**: 충격 직후 vs. 6개월 후 vs. 1년 후

### 정책 시뮬레이션
1. **정부 지원**: 긴급 자금 지원 시 효과
2. **보조금**: 국산화 보조금 효과 시뮬레이션
3. **규제 완화**: 규제 완화 시 공급망 회복 속도

---

## 11. 참고 문헌

1. 오석진, 이창민 (2020). "수출규제 3품목 수입동향 분석"
2. 한국은행 (2019). "일본의 수출규제 조치의 영향과 시사점"
3. 산업통상자원부 (2022). "소재·부품·장비 경쟁력강화 시행계획"
4. Inoue, H., & Todo, Y. (2019). "Firm-level propagation of shocks through supply-chain networks"
5. Barrot, J. N., & Sauvagnat, J. (2016). "Input specificity and the propagation of idiosyncratic shocks in production networks"
