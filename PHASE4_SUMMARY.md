# Phase 4: 제약 기반 최적 재배선 - 요약

**작성일:** 2026-01-19  
**상태:** ✅ 설계 완료, 구현 대기

---

## 🎯 Phase 4 핵심 개념

### 목표
관세 타격 시 **충격완충력(Buffer)**이 높은 안정적인 대체 거래선을 선정하여  
**경제적 손실을 최소화**하는 공급망 재배선

### 3단계 프로세스

```
1. 후보군 선정
   Phase 3 링크 예측 확률 상위 K개 추출
   ↓
2. 충격완충력 산출
   Buffer(v) = f(z_v) × 1/(TIS_v + ε)
   ↓
3. 최종 스코어링 & 선택
   Score = P(u,v) × Buffer(v) - Penalty_inv
```

---

## 🔬 핵심 수식

### 1. 충격완충력 (Buffer)
```
Buffer(v) = f(z_v) × 1/(TIS_v + ε)
```
- **f(z_v)**: 기업의 기초 체력 (매출 0.4 + 자산 0.3 + 영업이익 0.3)
- **TIS_v**: 관세 노출도 (높을수록 완충력 감소)
- **ε**: 분모가 0이 되는 것 방지 (1e-6)

### 2. 최종 스코어
```
Score_final(u,v) = P(u,v) × Buffer(v) - Penalty_inv(u,v)
```
- **P(u,v)**: Phase 3 링크 예측 확률 (0~1)
- **Buffer(v)**: 충격완충력
- **Penalty_inv**: 재고/용량 패널티

### 3. 재고/용량 패널티
```
Penalty_inv = α × RecipeMismatch + β × CapacityShortage
```
- **RecipeMismatch**: 레시피 불일치 (Cosine Distance)
- **CapacityShortage**: 용량 부족 비율
- **α, β**: 가중치 (기본값: 0.3, 0.2)

---

## 📊 입출력

### 입력
| 데이터 | 출처 | 형태 |
|--------|------|------|
| 링크 예측 확률 | Phase 3 | [N×N] |
| TIS 점수 | Phase 2 | [N] |
| 노드 임베딩 | Phase 2 | [N×32] |
| 생산함수 | Phase 1 | [N×33] |
| 재무 데이터 | Raw | 매출, 자산, 영업이익 |
| 단절 시나리오 | Input | 타격 대상 노드 리스트 |

### 출력
| 파일 | 내용 |
|------|------|
| `rewiring_map.pkl` | 소스→타겟 재배선 매핑 |
| `H_prime_rewired.npz` | 재배선된 네트워크 |
| `buffer_scores.npy` | 각 노드의 충격완충력 |
| `final_scores.npy` | 최종 스코어링 행렬 |

---

## 📁 파일 구조 (예정)

```
phase4/
├── main_phase4.py              # 메인 실행
├── evaluate_rewiring.py        # 평가
└── src/
    ├── rewiring_optimizer.py   # 재배선 알고리즘
    ├── buffer_calculator.py    # 충격완충력 계산
    ├── penalty_calculator.py   # 패널티 계산
    ├── constraint_checker.py   # 제약 조건 검증
    └── benchmarks.py           # Greedy, Random
```

---

## 🎯 벤치마크

| 방법 | 설명 |
|------|------|
| **TIS-Optimized** (제안) | P × Buffer - Penalty |
| **Greedy** | max P(u,v)만 고려 |
| **Random** | 무작위 선택 |

---

## 📈 평가 지표

1. **경제적 손실** (Phase 5에서 시뮬레이션)
2. **평균 Buffer** (선택된 타겟의 평균 충격완충력)
3. **평균 TIS** (선택된 타겟의 평균 관세 노출도)
4. **레시피 일치율** (레시피 유사도 평균)
5. **용량 적합률** (용량 충족 비율)

---

## 🔄 Phase 연결

```
Phase 3 → link_predictions.npy
          ↓
Phase 4 → rewiring_map.pkl
          H_prime_rewired.npz
          ↓
Phase 5 → 경제적 손실 비교
```

---

## ✅ 체크리스트

### Phase 4 실행 전
- [ ] Phase 3 완료 (link_predictions.npy)
- [ ] 재무 데이터 준비 (매출, 자산, 영업이익)
- [ ] 단절 시나리오 정의

### Phase 4 구현 시
- [ ] `rewiring_optimizer.py` 구현
- [ ] `buffer_calculator.py` 구현
- [ ] `penalty_calculator.py` 구현
- [ ] `constraint_checker.py` 구현
- [ ] `benchmarks.py` 구현
- [ ] `main_phase4.py` 구현
- [ ] 단위 테스트 작성

### Phase 4 완료 후
- [ ] 재배선 맵 생성 확인
- [ ] Buffer 분포 검증
- [ ] 벤치마크 비교
- [ ] Phase 5로 전달

---

## 💡 핵심 인사이트

1. **TIS 역수 사용**: 관세 노출도가 높을수록 완충력이 감소하도록 설계
2. **다중 제약 고려**: 레시피 일치, 용량 충족, 경제적 안정성 모두 반영
3. **해석 가능성**: 각 스코어 요소가 명확한 경제적 의미를 가짐
4. **확장 가능성**: 추가 제약 조건(거리, 신뢰도 등) 쉽게 추가 가능

---

**다음 단계:** Phase 4 구현 → Phase 5 시뮬레이션 설계
