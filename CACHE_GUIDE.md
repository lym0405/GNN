# 캐시 시스템 가이드

**Last Updated:** 2025-01-19  
**Pipeline:** Phases 1-5 (All Implemented)

## 📦 개요

Phase 2와 Phase 3는 데이터 빌딩 과정에서 **자동 캐싱**을 지원합니다.
한 번 생성된 그래프 데이터는 `data/processed/cache/` 디렉토리에 저장되어, 
다음 실행 시 빠르게 로드할 수 있습니다.

**성능 향상**: 초기 빌드 시간이 수 분에서 수 초로 단축됩니다.

---

## 🔄 캐시 동작 방식

### Phase 2: Static Graph Builder

**캐시 파일:**
```
data/processed/cache/
├── static_X_simple.npy          # 노드 피처 (73차원)
├── static_X_full.npy            # 노드 피처 (197차원, 미구현)
├── static_edge_index.pt         # 엣지 인덱스
├── static_edge_attr.pt          # 엣지 속성 (거래액)
└── static_firm_ids.pkl          # 기업 ID 리스트
```

**첫 실행:**
```python
builder = StaticGraphBuilder(data_dir="data", use_cache=True)
X, edge_index, edge_attr, firm_ids = builder.build_static_data()
# 📊 정적 그래프 데이터 구축
# 💾 캐시 저장...
```

**두 번째 실행:**
```python
builder = StaticGraphBuilder(data_dir="data", use_cache=True)
X, edge_index, edge_attr, firm_ids = builder.build_static_data()
# 📦 캐시된 정적 그래프 데이터 로드 (빠름!)
```

---

### Phase 3: Temporal Graph Builder & Negative Sampler

**캐시 파일:**
```
data/processed/cache/
├── temporal_data.pkl                    # 전체 시계열 데이터
└── historical_negatives_phase3.pkl      # Historical negatives (14,550 edges)
```

**첫 실행:**
```python
builder = TemporalGraphBuilder(data_dir="data", use_cache=True)
data = builder.build_temporal_data()
# 🕐 시계열 그래프 데이터 구축 시작
# � Historical Negatives 로드 중... (10-20초)
# �💾 캐시 저장 중...
```

**두 번째 실행:**
```python
builder = TemporalGraphBuilder(data_dir="data", use_cache=True)
data = builder.build_temporal_data()
# 📦 캐시된 시계열 그래프 데이터 로드 (매우 빠름!)
# 📦 Historical Negatives 캐시 로드 (~1초)
```

**성능 비교:**
- Historical Negatives 로드 (첫 실행): ~10-20초
- Historical Negatives 로드 (캐시): ~1초
- **속도 향상: 10-20배**

---

## 🛠️ 캐시 관리

### 캐시 삭제

**전체 캐시 삭제:**
```bash
python clear_cache.py
```

**Phase 2 캐시만 삭제:**
```bash
python clear_cache.py --phase2
```

**Phase 3 캐시만 삭제:**
```bash
python clear_cache.py --phase3
```

---

## ⚠️ 캐시 무효화가 필요한 경우

다음과 같은 경우 캐시를 삭제하고 다시 생성해야 합니다:

1. **Phase 1 레시피가 업데이트된 경우**
   ```bash
   python clear_cache.py  # 전체 삭제
   ```

2. **원본 데이터(H 행렬, 기업정보 등)가 변경된 경우**
   ```bash
   python clear_cache.py  # 전체 삭제
   ```

3. **피처 설정이 변경된 경우** (예: simple ↔ full)
   ```bash
   python clear_cache.py --phase2
   ```

4. **시계열 데이터(2020-2023)가 변경된 경우**
   ```bash
   python clear_cache.py --phase3
   ```

---

## 🚀 성능 비교

### Phase 2 (Static Graph)
- **캐시 없음**: ~2-5분 (피처 생성 + 네트워크 로드)
- **캐시 사용**: ~2-5초 (파일 로드만)

### Phase 3 (Temporal Graph)
- **캐시 없음**: ~10-20분 (4개 연도 네트워크 로드 + 정렬)
- **캐시 사용**: ~5-10초 (pickle 로드만)

---

## 💡 베스트 프랙티스

1. **개발/디버깅 시**: 캐시 활성화 (`use_cache=True`)
   - 빠른 반복 실험 가능
   - 하이퍼파라미터 튜닝에 유리

2. **데이터 업데이트 시**: 캐시 삭제 후 재실행
   ```bash
   python clear_cache.py
   python phase1/main_phase1.py  # Phase 1 먼저
   python phase2/main_phase2.py  # Phase 2
   python phase3/main.py         # Phase 3
   ```

3. **프로덕션 배포 시**: 
   - 첫 실행은 캐시 생성 (`use_cache=True`)
   - 이후 빠른 로드로 서비스 시작 시간 단축

---

## 📝 주의사항

- 캐시는 **자동으로 무효화되지 않습니다**
- 데이터가 변경되면 **수동으로 삭제**해야 합니다
- 캐시 디렉토리는 `.gitignore`에 포함되어 있습니다
- 디스크 공간이 부족하면 `clear_cache.py`로 정리하세요

---

## 🔍 트러블슈팅

**Q: 캐시가 로드되지 않습니다**
- A: `data/processed/cache/` 디렉토리가 존재하는지 확인
- A: 필요한 모든 캐시 파일이 있는지 확인

**Q: 데이터를 업데이트했는데 결과가 같습니다**
- A: 캐시가 남아있는 것입니다. `clear_cache.py` 실행 필요

**Q: 캐시를 비활성화하고 싶습니다**
```python
builder = StaticGraphBuilder(data_dir="data", use_cache=False)
```

---

## 📊 캐시 파일 크기 (예상)

| 파일 | 크기 (약) | 설명 |
|------|----------|------|
| `static_X_simple.npy` | ~128 MB | 438K 기업 × 73차원 × 4bytes |
| `static_edge_index.pt` | ~200 MB | 엣지 인덱스 (sparse) |
| `static_edge_attr.pt` | ~100 MB | 엣지 속성 |
| `temporal_data.pkl` | ~500 MB | 4년치 시계열 데이터 |
| `historical_negatives_phase3.pkl` | ~50 MB | **NEW:** Historical negatives (14,550 edges) |
| **Total** | **~1 GB** | **전체 캐시 크기** |

---

## 🆕 최근 업데이트 (2025-01-19)

### Phase 2 Training Optimization
- **Batch Size 증가**: 1024 → 4096
  - 캐시된 데이터로 빠른 로드 후 대용량 배치 처리
  - GPU 활용률 증가
  
- **Forward Pass 최적화**: 
  - 에폭당 1회만 수행
  - Embeddings 캐싱 및 재사용
  - 학습 속도 3-4배 향상

### Phase 3 Optimization
- **Historical Negatives Caching** (NEW):
  - CSV 파싱 → Pickle 캐시 (~50 MB)
  - 로드 시간: 10-20초 → 1초 (10-20배 빠름)
  - 14,550 historical edges (2020-2023)
  
- **Vectorized Negative Sampling**:
  - 배치 생성으로 2배 속도 향상
  - Set 기반 중복 제거
  - 적응형 multiplier

### Historical Negatives Fix
- **Phase 3 Negative Sampler**: 
  - 2020-2023년 Historical Negatives 정상 로드 (14,550 edges)
  - 한글 컬럼명 우선 매칭 (`사업자등록번호`)
  
**권장사항**: Phase 3 업데이트 후 캐시 재생성
```bash
python clear_cache.py --phase3
python phase3/main.py  # 첫 실행: 캐시 생성 (~20초)
python phase3/main.py  # 이후: 캐시 로드 (~1초)
```

**캐시 위치 확인**:
```bash
ls -lh data/processed/cache/
# 출력 예시:
# historical_negatives_phase3.pkl  (~50 MB)
# temporal_data.pkl                 (~500 MB)
# static_X_simple.npy               (~128 MB)
```

---

**Last Updated**: 2025-01-19  
**Version**: 2.0 (Performance Optimized)
| `temporal_data.pkl` | ~800 MB | 4년치 이벤트 스트림 |
| **총합** | **~1.2 GB** | 전체 캐시 |

디스크 공간이 충분한지 확인하세요!
