# Phase 4: Main 파일 수정 완료
**날짜:** 2024-01-20  
**상태:** ✅ **수정 완료**

---

## 🎯 문제점

기존 `phase4/main_phase4.py`가 **실제 존재하지 않는 파일과 컬럼**을 참조하여 실행 불가능했습니다.

### ❌ 기존 문제들:
1. **존재하지 않는 파일 참조**
   - `tis_scores_2024.csv` (실제로는 `tis_score_normalized.npy`)
   - `tg_{year}_filtered.csv` (동적 연도, 실제로는 `final_tg_2024_estimation.csv`)
   - `phase3/output/tis_scores_{year}.csv` (존재하지 않음)

2. **잘못된 컬럼명**
   - 실제 데이터의 컬럼명과 불일치

3. **과도한 설정 파일 의존**
   - YAML 설정 파일 요구 (불필요한 복잡성)

---

## ✅ 해결 방법

### 1. **실제 존재하는 파일 확인**

```bash
data/raw/
├── A_33.csv                           ✅
├── H_csr_model2.npz                   ✅
├── firm_to_idx_model2.csv             ✅
├── vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv  ✅
├── final_tg_2024_estimation.csv       ✅
├── asset_final_2024_6차.csv           ✅
├── export_estimation_value_final.csv  ✅
└── shock_after_P_v2.csv               ✅

data/processed/
├── disentangled_recipes.pkl           ✅ (Phase 1 출력)
├── recipes_dataframe.csv              ✅ (Phase 1 출력)
├── B_matrix.npy                       ✅ (Phase 1 출력)
├── node_embeddings_static.pt          ✅ (Phase 2 출력)
├── tis_score_normalized.npy           ✅ (Phase 2 출력)
├── train_edges.npy                    ✅ (Phase 2 출력)
└── test_edges.npy                     ✅ (Phase 2 출력)
```

### 2. **실제 컬럼명 확인**

| 파일 | 컬럼명 |
|------|--------|
| `final_tg_2024_estimation.csv` | `업체번호`, `tg_2024_final` |
| `asset_final_2024_6차.csv` | `업체번호`, `자산추정_2024` |
| `export_estimation_value_final.csv` | `업체번호`, `export_value` |
| `shock_after_P_v2.csv` | `업체번호`, `tis_score` (또는 첫 번째 값 컬럼) |
| `firm_to_idx_model2.csv` | `사업자등록번호`, (인덱스) |

---

## 🔧 수정 내용

### Config 클래스 (실제 파일 경로)
```python
class Config:
    # [Phase 1 출력]
    RECIPES = DATA_PROCESSED / "disentangled_recipes.pkl"  # ✅
    
    # [Phase 2 출력]
    TIS_SCORES = DATA_PROCESSED / "tis_score_normalized.npy"  # ✅
    
    # [Raw 데이터]
    H_MATRIX = DATA_RAW / "H_csr_model2.npz"  # ✅
    REVENUE = DATA_RAW / "final_tg_2024_estimation.csv"  # ✅
    ASSET = DATA_RAW / "asset_final_2024_6차.csv"  # ✅
    EXPORT = DATA_RAW / "export_estimation_value_final.csv"  # ✅
```

### 재무 데이터 로드 (실제 컬럼명)
```python
def load_financial_data(config: Config, firm_ids: list):
    # 매출
    df_rev = pd.read_csv(config.REVENUE)
    col_id = '업체번호' if '업체번호' in df_rev.columns else df_rev.columns[0]
    col_val = 'tg_2024_final' if 'tg_2024_final' in df_rev.columns else df_rev.columns[1]
    
    # 자산
    df_asset = pd.read_csv(config.ASSET)
    col_val = '자산추정_2024' if '자산추정_2024' in df_asset.columns else df_asset.columns[1]
    
    # 수출
    df_export = pd.read_csv(config.EXPORT)
    col_val = 'export_value' if 'export_value' in df_export.columns else df_export.columns[1]
```

---

## 📊 파이프라인 흐름

```
1. Phase 1-3 출력 로드
   ├─ 레시피 (disentangled_recipes.pkl)
   ├─ TIS 점수 (tis_score_normalized.npy)
   ├─ H 행렬 (H_csr_model2.npz)
   └─ 기업 ID (firm_to_idx_model2.csv)

2. 재무 데이터 로드
   ├─ 매출 (final_tg_2024_estimation.csv)
   ├─ 자산 (asset_final_2024_6차.csv)
   └─ 수출 (export_estimation_value_final.csv)

3. 충격완충력 계산
   Buffer = f(z_v) × 1/(TIS_v + ε)

4. 재배선 최적화
   Score = α×P(u,v) + β×Buffer(v) - γ×Penalty

5. 결과 저장
   ├─ buffer_scores.npy
   ├─ rewiring_map.pkl
   ├─ H_prime_rewired.npz
   └─ rewiring_report.csv
```

---

## 🚀 실행 방법

```bash
cd /Users/iyulim/Desktop/나이스/GNN

# Phase 1-3이 실행되었는지 확인
ls -la data/processed/
# disentangled_recipes.pkl, tis_score_normalized.npy 등이 있어야 함

# Phase 4 실행
python phase4/main_phase4.py
```

---

## ✅ 체크리스트

- [x] 실제 존재하는 파일만 참조
- [x] 실제 컬럼명 사용
- [x] Phase 1-3 출력 정확히 매핑
- [x] YAML 설정 파일 제거 (불필요)
- [x] 간결하고 명확한 구조
- [x] 에러 처리 및 로깅
- [x] Git 커밋 및 푸시

---

## 📝 주요 변경사항

### Before (문제)
```python
# ❌ 존재하지 않는 파일
tis_file = f'phase3/output/tis_scores_{year}.csv'

# ❌ 동적 연도 (불필요한 복잡성)
tg_file = f'data/processed/tg_{year}_filtered.csv'

# ❌ YAML 설정 파일 의존
config = load_config('config/phase4_config.yaml')
```

### After (해결)
```python
# ✅ 실제 존재하는 파일
TIS_SCORES = DATA_PROCESSED / "tis_score_normalized.npy"

# ✅ 고정된 파일명
REVENUE = DATA_RAW / "final_tg_2024_estimation.csv"

# ✅ 간단한 Config 클래스
class Config:
    # 명확한 경로 설정
    ...
```

---

## 📁 출력 파일

Phase 4 실행 후 생성되는 파일:
```
phase4/output/
├── buffer_scores.npy          # 충격완충력 점수
├── rewiring_map.pkl           # 재배선 매핑 (dict)
├── H_prime_rewired.npz        # 재배선된 네트워크 (sparse)
└── rewiring_report.csv        # 재배선 리포트 (csv)
```

---

## 🔄 다음 단계

1. **Phase 1-3 실행 확인**
   ```bash
   ls -la data/processed/
   # disentangled_recipes.pkl, tis_score_normalized.npy 등 확인
   ```

2. **Phase 4 실행**
   ```bash
   python phase4/main_phase4.py
   ```

3. **Phase 5 (Shock Simulation)**
   - Phase 4 출력(H', buffer) 사용
   - 충격 전파 시뮬레이션
   - 원본 vs 재배선 비교

---

**상태:** ✅ **Phase 4 Main 파일 수정 완료**  
**커밋:** `3768596 - fix: Phase 4 메인 파일 완전 재작성`

---

**작성:** 2024-01-20  
**작성자:** GNN Supply Chain Team
