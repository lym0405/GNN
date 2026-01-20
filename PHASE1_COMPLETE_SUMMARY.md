# Phase 1: Zero-Shot Inventory Module - 완료 요약
**날짜:** 2024-01-20  
**상태:** ✅ **완성**

---

## 🎯 목표

기업별 생산함수(33차원 레시피)를 추정하되, **단순 산업코드 매칭이 아닌 Zero-Shot Inventory Module** 구현

---

## ✅ 구현 완료 기능

### 1️⃣ **ProductMatcher** (상품 매칭)
**파일:** `phase1/src/product_matcher.py`

**기능:**
- TF-IDF 기반 텍스트 유사도 계산
- `주요상품목록` 필드에서 상품 키워드 추출
- `IO상품_다중_대분류_코드` 활용한 다중 상품 매칭
- Top-K 상품 선택

**핵심 메서드:**
```python
class ProductMatcher:
    def match_product(self, product_text: str, top_k: int = 3)
        # TF-IDF 유사도로 상위 K개 IO 상품 매칭
        
    def batch_match(self, df_firms, col_product_text, col_multi_code)
        # 전체 기업에 대해 배치 매칭
```

**예시:**
```
기업의 주요상품목록: "철강재, 강판, 후판, 철근"
→ 매칭 결과: 
  1. "1차 철강" (유사도: 0.85)
  2. "철강 1차 제품" (유사도: 0.78)
  3. "금속 제품" (유사도: 0.42)
```

---

### 2️⃣ **AttentionDisentangler** (레시피 분리)
**파일:** `phase1/src/attention_disentangler.py`

**기능:**
- Query-Key Attention 메커니즘으로 공급자-구매자 매칭
- 다중 상품 기업의 레시피 분리 (Disentangle)
- Temperature 기반 Softmax (sharp vs smooth)
- Prior (B 행렬) + Attention 가중 결합

**핵심 알고리즘:**

```python
# Query: 구매자 u의 상품 벡터 (예: [1, 0, 0, ...])
# Key: 공급자 v의 레시피 벡터 (예: [0.3, 0.5, 0.2, ...])

# 1. Attention Score 계산
attention_scores = softmax(Q @ K^T / temperature)

# 2. 거래 금액으로 가중치
weighted_recipes = attention_scores * H[u, v] * B[v]

# 3. Prior와 결합
final_recipe = alpha * attention_recipe + (1-alpha) * prior_recipe
```

**하이퍼파라미터:**
- `temperature`: 0.8 (낮을수록 sharp, 높을수록 smooth)
- `alpha`: 0.7 (Attention 신뢰도, 0.0=Prior만, 1.0=Attention만)

---

### 3️⃣ **통합 파이프라인**
**파일:** `phase1/main_phase1.py`

**실행 흐름:**
```
1. 데이터 로드
   ├─ IO 테이블 (33x33)
   ├─ H 행렬 (438K x 438K, sparse)
   ├─ 기업 정보 (주요상품목록, IO상품_다중_분류_코드)
   └─ 매출 데이터

2. B 행렬 생성 (BMatrixGenerator)
   └─ 기업별 기본 레시피 (산업코드 기반)

3. [NEW] Attention 기반 레시피 추정
   ├─ ProductMatcher로 상품 매칭
   ├─ AttentionDisentangler로 레시피 분리
   └─ 다중 상품 처리

4. 저장
   ├─ disentangled_recipes.pkl
   ├─ recipes_dataframe.csv
   └─ recipe_validation_report.csv
```

---

## 🔬 기술적 세부사항

### Query-Key Attention의 작동 원리

**시나리오:** 자동차 제조사 A가 철강사 B와 전자부품사 C로부터 구매

```
기업 A (자동차):
  - 주요상품: ["자동차", "완성차"]
  - Query 벡터: [0, 0, ..., 1(자동차), ..., 0]

공급자 B (철강):
  - 레시피: [0.5(1차 철강), 0.3(금속), 0.2(기타)]
  - H[A, B] = 1억원 (구매금액)

공급자 C (전자부품):
  - 레시피: [0.8(전자부품), 0.2(기타)]
  - H[A, C] = 5천만원

Attention 계산:
  - Score(A, B) = softmax(Q_A · K_B / 0.8) = 0.85
  - Score(A, C) = softmax(Q_A · K_C / 0.8) = 0.15

최종 레시피 (A):
  - 1차 철강: 0.85 * 1억 * 0.5 = 0.425
  - 전자부품: 0.15 * 5천만 * 0.8 = 0.06
  ... (정규화)
```

---

## 📊 출력 데이터

### 1. `disentangled_recipes.pkl`
```python
{
    'recipes': np.ndarray (N, 33),  # 기업별 레시피
    'firm_ids': List[str],          # 사업자등록번호
    'firm_products': Dict,          # 기업별 매칭된 상품 리스트
    'method': 'attention',
    'config': {
        'temperature': 0.8,
        'alpha': 0.7
    }
}
```

### 2. `recipes_dataframe.csv`
```csv
firm_id,IO_01,IO_02,...,IO_33
1234567890,0.25,0.15,...,0.05
9876543210,0.10,0.30,...,0.20
...
```

---

## 🆚 기존 방식 vs Zero-Shot Inventory Module

| 항목 | 기존 (단순 매칭) | Zero-Shot Module |
|------|-----------------|------------------|
| **산업코드** | IO상품_단일_대분류_코드만 | 주요상품목록 + 다중_분류_코드 |
| **다중 상품** | ❌ 불가능 | ✅ 가능 (Top-K) |
| **텍스트 분석** | ❌ 없음 | ✅ TF-IDF 유사도 |
| **Attention** | ❌ 없음 | ✅ Query-Key Matching |
| **레시피 분리** | ❌ 없음 | ✅ Disentangle |
| **정확도** | 낮음 (1:1 매칭) | 높음 (다중 상품 고려) |

**예시:**
```
기업: 현대제철
주요상품: "철강재, 강판, 후판, 철근, 선재"

[기존]
  → IO상품_단일_대분류_코드: "1차 철강" 하나만

[Zero-Shot]
  → Top-3 매칭: 
     1. "1차 철강" (0.85)
     2. "철강 1차 제품" (0.78)
     3. "금속 제품" (0.42)
  → Attention으로 공급자와 매칭하여 최종 레시피 생성
```

---

## 🔧 핵심 코드 스니펫

### ProductMatcher 사용법
```python
from phase1.src.product_matcher import ProductMatcher, create_io_product_dict

# IO 딕셔너리 생성
io_dict = create_io_product_dict("data/raw/A_33.csv")

# 매처 초기화
matcher = ProductMatcher(io_dict)

# 단일 기업 매칭
products = matcher.match_product("철강재, 강판, 후판", top_k=3)
# 결과: [("1차 철강", 0.85), ("철강 1차 제품", 0.78), ...]

# 배치 매칭
firm_products = matcher.batch_match(
    df_firms=firm_info,
    col_product_text='주요상품목록',
    col_multi_code='IO상품_다중_대분류_코드',
    use_multi_code=True,
    top_k=3
)
```

### AttentionDisentangler 사용법
```python
from phase1.src.attention_disentangler import create_disentangled_recipes

recipes = create_disentangled_recipes(
    H_matrix=H_sparse,
    B_matrix=B_matrix,
    firm_products=firm_products,
    firm_ids=firm_ids,
    method='attention',
    temperature=0.8,
    alpha=0.7
)
# 결과: (N, 33) 레시피 행렬
```

---

## 🚀 실행 방법

```bash
cd /Users/iyulim/Desktop/나이스/GNN
python phase1/main_phase1.py
```

**필요한 데이터:**
- `data/raw/A_33.csv` (IO 테이블)
- `data/raw/H_csr_model2.npz` (거래 네트워크)
- `data/raw/firm_to_idx_model2.csv` (인덱스 매핑)
- `data/raw/vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv` (기업 정보)
- `data/raw/final_tg_2024_estimation.csv` (매출)

---

## 📈 성능 및 검증

### 예상 개선 효과
1. **다중 상품 기업 처리**: 한 기업이 여러 상품 생산 시 더 정확한 레시피
2. **텍스트 기반 매칭**: 산업코드만으로는 잡지 못한 상품 발견
3. **Attention 기반 분리**: 거래 패턴 기반 레시피 정제
4. **Prior 결합**: 산업 표준(B 행렬) + 실제 거래 조합

### 검증 방법
```python
from phase1.src.check_recipe import RecipeValidator

validator = RecipeValidator(recipes_dict)
validator.run_all_checks()
validator.export_report("recipe_validation_report.csv")
```

---

## 🔍 향후 개선 사항

1. **BERT 기반 텍스트 임베딩**: TF-IDF → BERT로 업그레이드
2. **Transformer Attention**: 단순 Query-Key → Multi-Head Attention
3. **시계열 학습**: 2020-2024년 데이터로 동적 레시피 학습
4. **검증 강화**: Ground Truth와 비교 (알려진 기업 레시피)

---

## 📝 Git 커밋 이력

```
73ae2e5 - feat: Phase 1 Zero-Shot Inventory Module 완성
  - 텍스트 유사도 기반 상품 매칭 (ProductMatcher)
  - Query-Key Attention 메커니즘 (AttentionDisentangler)
  - 다중 상품 레시피 분리 (Disentangle)
```

---

## ✅ 체크리스트

- [x] ProductMatcher 구현 (TF-IDF 유사도)
- [x] AttentionDisentangler 구현 (Query-Key Attention)
- [x] 다중 상품 처리 (Top-K)
- [x] 주요상품목록 텍스트 분석
- [x] IO상품_다중_분류_코드 활용
- [x] main_phase1.py 통합
- [x] Git 커밋 및 푸시
- [x] 문서화 (본 파일)

---

**상태:** ✅ **Phase 1 Zero-Shot Inventory Module 완성**  
**다음 단계:** Phase 2 (정적 그래프 임베딩)에서 이 레시피를 Node Feature로 사용

---

**작성:** 2024-01-20  
**작성자:** GNN Supply Chain Team
