# Phase 4: Constrained Rewiring Strategy

**작성일:** 2026-01-19  
**Phase 4 목표:** TIS-Optimized 공급망 재배선

---

## 🎯 Phase 4 개요

### 목표
Phase 3의 링크 예측 결과와 TIS를 결합하여 **충격완충력(Buffer)**을 산출하고,  
이를 기준으로 관세 타격 시 안정적인 대체 거래선을 선정

### 핵심 개념
- **충격완충력(Buffer)**: 기업의 기초 체력 ÷ 관세 노출도
- **최적 재배선**: 거래 확률 × 완충력 - 재고/용량 패널티
- **경제적 손실 최소화**: 타격량을 최소화하는 재배선 선택

---

## 📐 Phase 4 로직

### 1️⃣ **후보군 선정**

Phase 3에서 예측된 링크 확률 상위 기업 추출

```python
# Phase 3 출력: P(u, v) - 링크 예측 확률
candidates = get_top_k_predictions(
    link_probs,
    k=100,  # 상위 100개 후보
    threshold=0.5  # 최소 확률
)
```

**입력:**
- `link_probs`: Phase 3 링크 예측 확률 [N×N]
- `src_node`: 재배선 대상 소스 노드

**출력:**
- `candidate_targets`: 상위 K개 후보 타겟 노드

---

### 2️⃣ **충격완충력(Buffer) 산출**

$$
\text{Buffer}(v) = f(\mathbf{z}_v) \times \frac{1}{\text{TIS}_v + \epsilon}
$$

**구성 요소:**

1. **$f(\mathbf{z}_v)$: 기업의 기초 체력**
   - Phase 2 임베딩 벡터 $\mathbf{z}_v$에서 추출
   - 매출 규모, 영업이익, 자산 등 반영
   
   ```python
   # 기초 체력 = 정규화된 재무 지표
   f_z = normalize(
       revenue[v] * 0.4 + 
       assets[v] * 0.3 + 
       operating_profit[v] * 0.3
   )
   ```

2. **$\text{TIS}_v$: 관세 노출도**
   - 높을수록 관세 타격에 취약
   - Phase 2에서 정규화된 TIS 점수 사용
   
   ```python
   # TIS가 높을수록 완충력 감소
   tis_penalty = 1.0 / (TIS[v] + epsilon)
   ```

**최종 충격완충력:**
```python
Buffer[v] = f_z[v] * tis_penalty[v]
```

---

### 3️⃣ **최종 스코어링**

$$
\text{Score}_{\text{final}}(u,v) = P(u,v) \times \text{Buffer}(v) - \text{Penalty}_{\text{inv}}(u,v)
$$

**구성 요소:**

1. **$P(u,v)$: 링크 예측 확률** (Phase 3)
   - 0~1 사이 값
   - 거래 성사 가능성

2. **$\text{Buffer}(v)$: 충격완충력**
   - 타겟 기업의 안정성
   - 관세 타격 흡수 능력

3. **$\text{Penalty}_{\text{inv}}(u,v)$: 재고/용량 패널티**
   
   ```python
   # 레시피 불일치 패널티
   recipe_mismatch = cosine_distance(
       recipe[u],  # Phase 1 생산함수
       recipe[v]
   )
   
   # 용량 부족 패널티 (매출 기준)
   capacity_shortage = max(0, 
       required_volume[u] - available_capacity[v]
   ) / required_volume[u]
   
   # 최종 패널티
   Penalty_inv = alpha * recipe_mismatch + beta * capacity_shortage
   ```

**최종 점수:**
```python
Score_final[u, v] = (
    link_prob[u, v] * Buffer[v] 
    - penalty_inv[u, v]
)
```

---

### 4️⃣ **재배선 선택**

각 소스 노드 $u$에 대해 최고 점수를 가진 타겟 선택

```python
for src_node in disrupted_nodes:
    # 후보군에서 최고 점수 선택
    best_target = argmax(Score_final[src_node, candidates])
    
    # 재배선 매핑 저장
    rewiring_map[src_node] = best_target
```

**제약 조건:**
- 타겟 노드의 총 수용 용량 초과 방지
- 레시피 불일치가 임계값 이상인 경우 제외
- 최소 Buffer 임계값 미달 시 제외

---

## 📊 Phase 4 입출력

### **입력**

| 데이터 | 출처 | 형태 |
|--------|------|------|
| **노드 임베딩** | Phase 2 | `node_embeddings_static.pt` [N×32] |
| **링크 예측 확률** | Phase 3 | `link_predictions.npy` [N×N] |
| **TIS 점수** | Phase 2 | `tis_score_normalized.npy` [N] |
| **생산함수(레시피)** | Phase 1 | `disentangled_recipes.pkl` [N×33] |
| **재무 데이터** | Raw | 매출, 자산, 영업이익 |
| **단절 시나리오** | 입력 | 타격 대상 노드 리스트 |

### **출력**

| 파일 | 내용 | 형태 |
|------|------|------|
| **`rewiring_map.pkl`** | 소스→타겟 재배선 매핑 | Dict[int, int] |
| **`buffer_scores.npy`** | 각 노드의 충격완충력 | [N] |
| **`final_scores.npy`** | 최종 스코어링 행렬 | [N×N] |
| **`H_prime_rewired.npz`** | 재배선된 네트워크 | Sparse [N×N] |

---

## 🎯 Phase 4 알고리즘

### **Constrained Rewiring Algorithm**

```python
def phase4_rewiring(
    node_embeddings,      # Phase 2
    link_probs,           # Phase 3
    tis_scores,           # Phase 2
    recipes,              # Phase 1
    financial_data,       # Raw
    disrupted_nodes,      # Input
    top_k=100,            # 후보군 크기
    alpha=0.3,            # 레시피 패널티 가중치
    beta=0.2              # 용량 패널티 가중치
):
    """
    제약 기반 최적 재배선
    """
    
    # Step 1: 충격완충력 계산
    buffers = compute_buffer(
        node_embeddings,
        financial_data,
        tis_scores
    )
    
    # Step 2: 각 단절 노드에 대해
    rewiring_map = {}
    
    for src in disrupted_nodes:
        # Step 2.1: 후보군 선정
        candidates = get_top_k_candidates(
            link_probs[src],
            k=top_k
        )
        
        # Step 2.2: 최종 스코어 계산
        scores = []
        for tgt in candidates:
            # 링크 확률
            p_uv = link_probs[src, tgt]
            
            # 충격완충력
            buffer = buffers[tgt]
            
            # 재고/용량 패널티
            penalty = compute_penalty(
                recipes[src],
                recipes[tgt],
                financial_data[src],
                financial_data[tgt],
                alpha, beta
            )
            
            # 최종 점수
            score = p_uv * buffer - penalty
            scores.append((tgt, score))
        
        # Step 2.3: 최고 점수 선택
        best_target = max(scores, key=lambda x: x[1])[0]
        rewiring_map[src] = best_target
    
    return rewiring_map


def compute_buffer(embeddings, financial, tis):
    """
    충격완충력 계산
    
    Buffer(v) = f(z_v) × 1/(TIS_v + ε)
    """
    # 기초 체력
    f_z = normalize(
        financial['revenue'] * 0.4 +
        financial['assets'] * 0.3 +
        financial['operating_profit'] * 0.3
    )
    
    # TIS 페널티
    epsilon = 1e-6
    tis_penalty = 1.0 / (tis + epsilon)
    
    # 충격완충력
    buffer = f_z * tis_penalty
    
    return buffer


def compute_penalty(recipe_u, recipe_v, fin_u, fin_v, alpha, beta):
    """
    재고/용량 패널티 계산
    """
    # 레시피 불일치 (Cosine Distance)
    recipe_mismatch = 1.0 - cosine_similarity(recipe_u, recipe_v)
    
    # 용량 부족 (매출 기준)
    required = fin_u['required_volume']
    available = fin_v['available_capacity']
    capacity_shortage = max(0, required - available) / required
    
    # 최종 패널티
    penalty = alpha * recipe_mismatch + beta * capacity_shortage
    
    return penalty
```

---

## 📈 Phase 4 평가

### **벤치마크 모델**

1. **Greedy Baseline**
   - 링크 확률만 고려
   - TIS/Buffer 무시
   
   ```python
   greedy_target = argmax(link_probs[src])
   ```

2. **Random Baseline**
   - 무작위 후보 선택
   
   ```python
   random_target = random.choice(candidates)
   ```

3. **TIS-Optimized (제안 방법)**
   - 링크 확률 × Buffer - Penalty

### **평가 지표**

| 지표 | 설명 | 계산 |
|------|------|------|
| **경제적 손실** | 재배선 후 총 손실액 | Phase 5에서 시뮬레이션 |
| **평균 Buffer** | 선택된 타겟의 평균 충격완충력 | `mean(Buffer[targets])` |
| **평균 TIS** | 선택된 타겟의 평균 관세 노출도 | `mean(TIS[targets])` |
| **레시피 일치율** | 레시피 유사도 평균 | `mean(cosine_sim(u, v))` |
| **용량 적합률** | 용량 충족 비율 | `sum(capacity_ok) / N` |

---

## 🗂️ Phase 4 파일 구조

```
GNN/
└── phase4/
    ├── README.md                      # Phase 4 설명서
    ├── STRUCTURE.txt                  # Phase 4 구조 문서
    ├── requirements.txt               # Python 의존성
    │
    ├── main_phase4.py                 # Phase 4 메인 실행 파일
    ├── evaluate_rewiring.py           # 재배선 평가 스크립트
    │
    └── src/
        ├── rewiring_optimizer.py      # 재배선 최적화 알고리즘
        ├── buffer_calculator.py       # 충격완충력 계산
        ├── penalty_calculator.py      # 재고/용량 패널티 계산
        ├── constraint_checker.py      # 제약 조건 검증
        └── benchmarks.py              # Greedy, Random 벤치마크
```

---

## 🔄 Phase 3 → Phase 4 → Phase 5 연결

```
Phase 3 (Link Prediction)
   ↓
   출력: link_predictions.npy [N×N]
   ↓
Phase 4 (Constrained Rewiring)
   ↓
   입력: link_predictions.npy + TIS + recipes + financial
   처리: Buffer 계산 → 최종 스코어링 → 재배선 선택
   출력: rewiring_map.pkl, H_prime_rewired.npz
   ↓
Phase 5 (Resilience Simulation)
   ↓
   입력: H_original + H_prime_rewired
   처리: 충격 전파 시뮬레이션
   평가: 경제적 손실 비교 (원본 vs 재배선)
```

---

## 🎯 핵심 수식 정리

### 1. **충격완충력**
$$
\text{Buffer}(v) = f(\mathbf{z}_v) \times \frac{1}{\text{TIS}_v + \epsilon}
$$

### 2. **최종 스코어**
$$
\text{Score}_{\text{final}}(u,v) = P(u,v) \times \text{Buffer}(v) - \text{Penalty}_{\text{inv}}(u,v)
$$

### 3. **재고/용량 패널티**
$$
\text{Penalty}_{\text{inv}}(u,v) = \alpha \cdot \text{RecipeMismatch}(u,v) + \beta \cdot \text{CapacityShortage}(u,v)
$$

---

## ✅ Phase 4 실행 체크리스트

- [ ] Phase 3 완료 (link_predictions.npy 생성)
- [ ] 재무 데이터 로드 (매출, 자산, 영업이익)
- [ ] 단절 시나리오 정의 (타격 대상 노드)
- [ ] 충격완충력 계산 (Buffer)
- [ ] 최종 스코어링 (Score_final)
- [ ] 재배선 매핑 생성 (rewiring_map)
- [ ] 벤치마크 비교 (Greedy, Random)
- [ ] Phase 5로 전달 (H_prime_rewired.npz)

---

**작성자:** Phase 4 기획  
**목표:** TIS-Optimized 공급망 재배선  
**다음 단계:** Phase 5 (Resilience Simulation)
