# Phase 1: Production Recipe Estimation

기업별 생산함수(33차원 레시피) 추정 파이프라인

## 📋 개요

각 기업이 33개 산업의 중간재를 얼마나 사용하는지 추정합니다.

- **입력**: IO 테이블(33×33), 거래 네트워크(H), 기업 정보, 매출
- **출력**: 기업별 33차원 레시피 벡터
- **방법**: BMatrixGenerator + ZeroShotInventoryModule

## 🚀 빠른 시작

### 1. 데이터 준비

`data/raw/` 폴더에 다음 파일을 배치:

```
data/raw/
├── A_33.csv                                                      # IO 테이블
├── H_csr_model2.npz                                              # 거래 네트워크
├── firm_to_idx_model2.csv                                        # 기업 인덱스
├── vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv  # 기업 정보
└── tg_2024_filtered.csv                                          # 매출 데이터
```

### 2. 환경 설정

```bash
pip install numpy pandas scipy matplotlib
```

### 3. 실행

```bash
python main_phase1.py
```

### 4. 출력 확인

```
data/processed/
├── disentangled_recipes.pkl         # 레시피 (pickle)
├── recipes_dataframe.csv            # 레시피 (CSV)
├── B_matrix.npy                     # B 행렬
└── recipe_validation_report.csv    # 검증 리포트
```

## 📂 파일 구조

### 핵심 모듈

- **`src/b_matrix_generator.py`**: B 행렬 생성
  - 국가 IO 테이블을 기업 수준으로 변환
  - 산업별 기술계수를 기업에 할당

- **`src/inventory_module.py`**: 레시피 추정
  - 거래 네트워크(H)와 B 행렬을 결합
  - 3가지 추정 방법: weighted, simple, bayesian

- **`src/check_recipe.py`**: 레시피 검증
  - 기본 속성 체크 (NaN, Inf, 음수, 행 합)
  - 통계적 속성 분석
  - 다양성 및 이상치 탐지

- **`src/debug_deep_dive.py`**: 디버깅 도구
  - 특정 기업 상세 분석
  - 기업 간 비교
  - 시각화

### 실행 파일

- **`main_phase1.py`**: 메인 파이프라인
  - 데이터 로드 → B 행렬 생성 → 레시피 추정 → 검증

## 🔧 사용법

### 기본 실행

```bash
python main_phase1.py
```

### 레시피 검증만 실행

```bash
python src/check_recipe.py data/processed/disentangled_recipes.pkl
```

### 특정 기업 분석

```bash
python src/debug_deep_dive.py data/processed/disentangled_recipes.pkl --firm <사업자번호>
```

### 기업 비교

```bash
python src/debug_deep_dive.py data/processed/disentangled_recipes.pkl --compare <firm1> <firm2> <firm3>
```

### 랜덤 샘플링

```bash
python src/debug_deep_dive.py data/processed/disentangled_recipes.pkl --random 5
```

## ⚙️ 설정

`main_phase1.py`의 `Config` 클래스에서 설정 변경 가능:

```python
class Config:
    # 추정 방법
    ESTIMATION_METHOD = 'weighted'  # 'weighted', 'simple', 'bayesian'
    
    # 매출 가중치 사용 여부
    USE_REVENUE_WEIGHTING = True
    
    # 배치 크기 (대용량 데이터용)
    BATCH_SIZE = 10000
```

## 📊 레시피 추정 방법

### 1. Weighted (기본값, 권장)

거래 금액으로 공급자의 레시피를 가중 평균:

```
Recipe[i, k] = Σ_j (H[i,j] × B[j,k]) / Σ_j H[i,j]
```

**장점**: 거래 규모를 반영, 가장 현실적
**단점**: H 행렬이 sparse하면 정보 부족

### 2. Simple

공급자들의 레시피를 동일 가중치로 평균:

```
Recipe[i, k] = mean(B[suppliers, k])
```

**장점**: 계산 빠름
**단점**: 거래 규모 무시

### 3. Bayesian

B를 Prior로, Weighted를 Likelihood로:

```
Recipe[i] = α × B[i] + (1-α) × WeightedRecipe[i]
```

**장점**: B의 산업 지식과 H의 실제 거래 균형
**단점**: α 튜닝 필요

## ✅ 검증 기준

레시피가 다음 조건을 만족해야 함:

1. **NaN/Inf 없음**: 모든 값이 유효한 실수
2. **음수 없음**: 모든 값 ≥ 0
3. **행 합 = 1**: 각 기업의 33차원 벡터 합이 1
4. **적절한 다양성**: 평균 사용 산업 수 5~15개
5. **극단 집중 최소화**: 한 산업이 90% 이상인 케이스 < 5%

## 🐛 문제 해결

### "FileNotFoundError: A_33.csv"

→ `data/raw/` 폴더에 필요한 데이터 파일 배치

### "MemoryError"

→ `Config.BATCH_SIZE`를 줄이거나, 서버에서 실행

### "극단 집중 기업이 너무 많음"

→ `Config.ESTIMATION_METHOD = 'bayesian'`으로 변경

### "매출 점유율 계산 실패"

→ 매출 데이터의 컬럼명 확인 (tg_2024_final, revenue 등)

## 📈 다음 단계

Phase 1 완료 후:

1. `data/processed/disentangled_recipes.pkl` 생성 확인
2. `recipe_validation_report.csv`로 품질 체크
3. **Phase 2**로 진행: 이 레시피를 피처로 사용하여 GNN 학습

## 📚 참고

- 논문: "Zero-Shot Production Function Estimation using Input-Output Tables"
- 데이터: 한국은행 산업연관표 (33부문)
- 기술: Sparse Matrix, Dictionary-based Recipe Storage

## 👥 기여

버그 제보 및 개선 제안 환영합니다!
