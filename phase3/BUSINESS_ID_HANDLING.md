# 사업자등록번호 처리 방식

## 개요

Phase 3 negative sampler는 한국의 **사업자등록번호**(10자리 숫자)를 기반으로 기업을 식별합니다.

## 사업자등록번호란?

- **형식:** 10자리 숫자 (예: `1234567890`)
- **용도:** 한국 국세청이 사업자에게 부여하는 고유 식별번호
- **구조:** XXX-XX-XXXXX (3-2-5 자리로 구분되나, 데이터에서는 구분자 없이 10자리로 저장)

## 현재 데이터 형식

### 익명화된 형태
실제 프로젝트 데이터는 개인정보 보호를 위해 **익명화된 형태**로 저장됩니다:

```
firm_000000
firm_000001
firm_000002
...
firm_000499
```

- **패턴:** `firm_XXXXXX` (firm_ + 일련번호)
- **목적:** 실제 사업자등록번호 노출 방지
- **매핑:** `firm_to_idx_model2.csv`에서 익명화 ID → 내부 인덱스 변환

### 실제 사업자등록번호 (향후 사용 가능)
실제 사업자등록번호를 사용하는 경우:

```
1234567890
2345678901
3456789012
```

- **패턴:** 10자리 숫자
- **검증:** 정규식 `^\d{10}$`로 형식 확인

## 코드 처리 방식

### 1. 문자열 통일 처리
모든 사업자등록번호를 **문자열(str)**로 통일하여 처리:

```python
# firm_to_idx 매핑 생성 시
firm_to_idx = dict(zip(
    firm_to_idx_df['사업자등록번호'].astype(str),  # ← str로 명시적 변환
    firm_to_idx_df['idx']
))

# 네트워크 파일 읽을 때
src_firm = str(row[src_col])  # ← str로 명시적 변환
dst_firm = str(row[dst_col])
```

**이유:**
- CSV에서 10자리 숫자가 `int`로 자동 읽힐 수 있음
- 문자열로 통일하면 익명화/실제 모두 동일하게 처리 가능

### 2. 자동 형식 감지
데이터 로딩 시 자동으로 형식을 감지하고 로깅:

```python
def get_business_id_format(sample_ids: List[str]) -> str:
    """ID 형식 감지: 'real', 'anonymized', 'unknown'"""
    if re.match(r'^\d{10}$', sample_ids[0]):
        return 'real'  # 실제 10자리 숫자
    elif re.match(r'^firm_\d+$', sample_ids[0]):
        return 'anonymized'  # 익명화 형태
    else:
        return 'unknown'
```

### 3. 검증 함수
필요 시 ID 형식을 검증:

```python
def validate_business_id(business_id: str) -> bool:
    """사업자등록번호 형식 검증"""
    # 10자리 숫자 OR firm_XXXXXX
    return (
        re.match(r'^\d{10}$', business_id) or 
        re.match(r'^firm_\d+$', business_id)
    )
```

## 로그 출력 예시

```
✓ Firm-to-Index 매핑: 500개 기업
✓ ID 형식: 익명화된 형태 (firm_XXXXXX) (예: firm_000000)
✓ 2020년: 3,719개 추가 (누적: 3,719개)
✓ 2021년: 3,661개 추가 (누적: 7,380개)
```

또는 실제 사업자등록번호 사용 시:

```
✓ Firm-to-Index 매핑: 500개 기업
✓ ID 형식: 실제 사업자등록번호 (10자리 숫자) (예: 1234567890)
```

## 데이터 파일 구조

### firm_to_idx_model2.csv
```csv
사업자등록번호,idx
firm_000000,0
firm_000001,1
firm_000002,2
```

또는 실제 사업자등록번호:
```csv
사업자등록번호,idx
1234567890,0
2345678901,1
3456789012,2
```

### posco_network_capital_consumergoods_removed_YYYY.csv
```csv
Unnamed: 0,Unnamed: 1,사업자등록번호,거래처사업자등록번호,총공급금액
firm_000461,firm_000185,firm_000461,firm_000185,151450908.29
firm_000077,firm_000441,firm_000077,firm_000441,119646028.32
```

## 주의사항

### 1. CSV 읽기 시 dtype 지정
10자리 숫자가 과학적 표기법으로 변환되는 것을 방지:

```python
# 올바른 방법
df = pd.read_csv('file.csv', dtype={'사업자등록번호': str})

# 또는 읽은 후 변환
df['사업자등록번호'] = df['사업자등록번호'].astype(str)
```

### 2. 앞자리 0 보존
실제 사업자등록번호가 0으로 시작하는 경우 (예: `0123456789`):
- `int`로 읽으면 → `123456789` (앞자리 0 손실)
- `str`로 읽으면 → `0123456789` (보존됨)

### 3. 매칭 일관성
- `firm_to_idx` 매핑: str로 저장
- 네트워크 파일 읽기: str로 변환
- 두 소스의 dtype이 일치해야 매칭 성공

## 테스트

### 자동 테스트
```bash
cd phase3
python test_historical_negatives.py
```

### 수동 확인
```python
import pandas as pd

# firm_to_idx 확인
df = pd.read_csv('data/raw/firm_to_idx_model2.csv')
print(df.head())
print(df['사업자등록번호'].dtype)  # object (문자열)인지 확인

# 네트워크 파일 확인
df = pd.read_csv('data/raw/posco_network_2020.csv')
print(df['사업자등록번호'].dtype)
```

## 향후 개선 사항

1. **실제 사업자등록번호 사용 시:**
   - 체크섬 검증 추가 (사업자등록번호는 체크 디지트 포함)
   - 형식 오류 자동 수정 (하이픈 제거 등)

2. **대용량 데이터 처리:**
   - 현재는 `iterrows()` 사용 → 대용량 시 `itertuples()` 또는 벡터화 고려
   - 메모리 효율을 위한 청크 단위 처리

3. **데이터 품질 검증:**
   - 중복 사업자등록번호 체크
   - 유효하지 않은 형식 자동 필터링
   - 매핑 누락 통계 리포팅

## 참고 자료

- 국세청 사업자등록번호 안내: https://www.nts.go.kr/
- 사업자등록번호 체크 알고리즘: [별도 문서 참조]
