# Phase 3 Historical Negatives Fix

## Problem Diagnosis

The Phase 3 training was showing "Historical Negatives: 0" in the logs, meaning no historical negative samples were being loaded from past years (2020-2023).

## Root Cause

The issue was in the column name matching logic in `phase3/src/negative_sampler.py`:

1. **Actual data structure (after standardization):**
   - `firm_to_idx_model2.csv` has columns: `사업자등록번호` (firm ID) and `idx` (index)
   - Network files have columns: `사업자등록번호` (source) and `거래처사업자등록번호` (destination)

2. **Previous code logic:**
   - Was checking for `'Unnamed: 0'` first (which doesn't exist in the actual data)
   - Then falling back to `'firm_id'` (which also doesn't exist)
   - Never checked for `'사업자등록번호'` (the actual Korean column name)

3. **Result:**
   - The firm_to_idx mapping was never created
   - Historical negative loading failed silently
   - Returned empty set → "Historical Negatives: 0"

**Note:** The `firm_to_idx_model2.csv` file has been standardized to use `사업자등록번호` and `idx` columns for consistency.

## Solution

### 1. Standardized Column Names and Simplified Logic
After standardizing the `firm_to_idx_model2.csv` file to use consistent Korean column names, the code now uses a simple, direct approach:

```python
# 표준화된 컬럼명 사용: 사업자등록번호, idx
if '사업자등록번호' not in firm_to_idx_df.columns or 'idx' not in firm_to_idx_df.columns:
    logger.warning(
        f"⚠️  firm_to_idx 파일의 컬럼명이 올바르지 않습니다. "
        f"예상: ['사업자등록번호', 'idx'], 실제: {list(firm_to_idx_df.columns)}"
    )
    return historical_set

firm_to_idx = dict(zip(
    firm_to_idx_df['사업자등록번호'],
    firm_to_idx_df['idx']
))
```

This is much cleaner and more maintainable than the previous cascading if-elif chain.

### 2. Improved Logging
Enhanced year tracking and logging to show:
- Correctly tracked year (not just index)
- Number of edges added per year
- Cumulative count

```python
added_count = len(historical_set) - prev_count
logger.info(f"   ✓ {year}년: {added_count:,}개 추가 (누적: {len(historical_set):,}개)")
```

## Results

### Before Fix
```
Historical Negatives: 0
```

### After Fix
```
✓ Historical Negatives Loaded: 14,550
✓ 2020년: 3,642개 추가 (누적: 3,642개)
✓ 2021년: 3,108개 추가 (누적: 6,750개)
✓ 2022년: 3,891개 추가 (누적: 10,641개)
✓ 2023년: 3,909개 추가 (누적: 14,550개)
```

## Files Changed

1. **`phase3/src/negative_sampler.py`**
   - Fixed column name priority in `_load_historical_negatives()`
   - Improved year tracking and logging

2. **`phase3/test_historical_negatives.py`** (NEW)
   - Created test script to verify historical negative loading
   - Useful for debugging and validation

## Testing

Run the test script to verify:
```bash
cd phase3
python test_historical_negatives.py
```

Expected output:
- ✅ Historical negatives loaded: ~14,550
- Shows breakdown by year
- Displays sample negative pairs

## Impact

1. **Training Quality:** Phase 3 will now properly use historical negatives, leading to better model learning and generalization
2. **Sampling Strategy:** 50% historical + 50% random negatives (configurable)
3. **Data Utilization:** Leverages 4 years of historical network data (2020-2023)

## Next Steps

1. ✅ Fix applied and tested
2. ⏳ Run Phase 3 training with the fix and monitor:
   - Training loss convergence
   - Validation metrics (AUC, AP)
   - Training speed
3. ⏳ Compare results with/without historical negatives

## Notes

- **Data Standardization:** The `firm_to_idx_model2.csv` file has been updated to use standardized column names (`사업자등록번호`, `idx`)
- **Simplified Code:** Removed complex fallback logic for old column names, making the code cleaner and more maintainable
- **Better Error Messages:** If column names don't match, the error message now shows both expected and actual columns
- **Year-by-year Logging:** Enhanced progress logging for better debugging during data loading
