# 컬럼명 매핑 수정 완료 (2026.01.19)

## 🎯 핵심 문제
- IO 테이블(33개 산업)과 매칭 시 잘못된 컬럼 사용
- **KSIC 코드** 대신 **IO상품 코드**를 사용해야 함

## ✅ 수정된 컬럼명

### 1. 기업 정보 파일 (`vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv`)

**사용해야 할 컬럼:**
- `IO상품_단일_대분류_코드` ← **이것으로 33개 IO 산업과 매칭**

**사용하면 안 되는 컬럼:**
- ❌ `산업코드ID` (KSIC 코드)
- ❌ `KSIC_추출` (KSIC 코드)
- ❌ `IO산업_대분류_코드` (IO 산업 ≠ IO 상품)

### 2. 거래 네트워크 파일 (`posco_network_capital_consumergoods_removed_20XX.csv`)

**컬럼명:**
- `사업자등록번호` ← source (구매자)
- `거래처사업자등록번호` ← target (공급자)
- `총공급금액` ← 거래액
- `IO상품_단일_대분류_코드_사업자` ← 사업자 IO 산업
- `IO상품_단일_대분류_코드_거래처` ← 거래처 IO 산업

## 📝 수정된 파일 목록

### Phase 1
1. **`phase1/src/b_matrix_generator.py`** ✅
   - 변경: `col_sec` 찾기 로직 수정
   - 이전: `next(c for c in df_firm.columns if 'IO' in c and '대분류' in c)`
   - 이후: `next((c for c in df_firm.columns if c == 'IO상품_단일_대분류_코드'), ...)`

### Phase 3
2. **`phase3/src/temporal_graph_builder.py`** ✅
   - 변경: source/target 컬럼 찾기 로직 수정
   - 추가: 한글 컬럼명 우선 순위 (사업자등록번호, 거래처사업자등록번호)
   - 폴백: 영문 컬럼명 (더미 데이터용)

3. **`phase3/src/negative_sampler.py`** ✅
   - 변경: Historical negatives 로드 시 컬럼 찾기 로직 수정
   - 추가: 한글 컬럼명 우선 순위
   - 폴백: 영문 컬럼명 (더미 데이터용)

## 🔍 컬럼 찾기 우선순위

```python
# 1순위: 정확한 컬럼명 매칭
if '사업자등록번호' in df.columns:
    source_col = '사업자등록번호'
if '거래처사업자등록번호' in df.columns:
    target_col = '거래처사업자등록번호'

# 2순위: 부분 문자열 매칭
for col in df.columns:
    if '사업자' in col and '번호' in col and '거래처' not in col:
        source_col = col
        break

# 3순위: 영문 컬럼명 (더미 데이터용)
if 'source' in col.lower() or 'from' in col.lower():
    source_col = col
```

## 🎯 IO 상품 vs IO 산업 차이

### IO 상품 (사용해야 함)
- **`IO상품_단일_대분류_코드`**: 기업이 생산하는 **상품**의 IO 분류
- 33개 IO 테이블과 매칭됨
- 예: 자동차 제조업체 → 자동차 상품 생산

### IO 산업 (사용하면 안 됨)
- **`IO산업_대분류_코드`**: 기업이 속한 **산업**의 IO 분류
- KSIC 기반으로 매핑됨
- 예: 자동차 제조업체 → 제조업

## 📊 데이터 흐름

```
[기업 정보 파일]
├── 업체번호
├── 사업자등록번호
└── IO상품_단일_대분류_코드 ← 33개 IO 테이블과 매칭
        ↓
    [A_33.csv]
    33×33 IO 테이블
        ↓
    [B 행렬 생성]
    기업별 표준 레시피 (33차원)
```

```
[거래 네트워크 파일]
├── 사업자등록번호 (source)
├── 거래처사업자등록번호 (target)
├── 총공급금액
└── IO상품_단일_대분류_코드_사업자/거래처
        ↓
    [H 행렬]
    438K×438K 거래 네트워크
        ↓
    [레시피 추정]
    기업별 실제 레시피 (33차원)
```

## 🧪 테스트 방법

### 1. Phase 1 테스트
```bash
cd phase1
python main_phase1.py
```

**확인사항:**
- B 행렬이 정상 생성되는지
- `IO상품_단일_대분류_코드` 매칭이 잘 되는지
- 33개 산업 모두 커버되는지

### 2. Phase 3 테스트
```bash
cd phase3
python quick_test.py
```

**확인사항:**
- 시계열 네트워크 로드 시 `사업자등록번호`, `거래처사업자등록번호` 찾는지
- Historical negatives 생성 시 컬럼 매칭 잘 되는지

## 🚨 주의사항

1. **더미 데이터**: 영문 컬럼명 사용 (source, target, Unnamed:0/1)
2. **실제 데이터**: 한글 컬럼명 사용 (사업자등록번호, 거래처사업자등록번호)
3. **우선순위**: 한글 → 영문 순으로 체크하여 둘 다 호환

## ✅ 확인 완료

- [x] Phase 1: `IO상품_단일_대분류_코드` 사용
- [x] Phase 3 (temporal_graph_builder): 한글 컬럼명 우선
- [x] Phase 3 (negative_sampler): 한글 컬럼명 우선
- [x] 더미 데이터 호환성 유지 (영문 폴백)

---

**수정 완료일**: 2026년 1월 19일
**수정자**: AI Assistant
**이슈**: 회사 노트북으로 옮기니 매핑 안됨
**원인**: KSIC 코드 대신 IO상품 코드 사용해야 함
