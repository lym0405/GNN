# GNN Project - Python Files Tree

## 📂 Project Python Files (Excluding .venv)

```
GNN/
├── clear_cache.py
│
├── phase1/
│   ├── generate_dummy_data.py
│   ├── main_phase1.py
│   └── src/
│       ├── b_matrix_generator.py
│       ├── check_recipe.py
│       ├── debug_deep_dive.py
│       └── inventory_module.py
│
├── phase2/
│   ├── main_phase2.py
│   ├── test_phase2.py
│   └── src/
│       ├── GraphSAGE.py
│       ├── graph_builder.py
│       ├── loss.py
│       ├── sampler.py
│       └── trainer.py
│
├── phase3/
│   ├── evaluate_comprehensive.py
│   ├── generate_temporal_networks.py
│   ├── main.py
│   ├── main_old.py
│   ├── quick_test.py
│   ├── test.py
│   └── src/
│       ├── benchmarks.py
│       ├── graphseal.py
│       ├── hybrid_trainer.py
│       ├── link_predictor.py
│       ├── loss.py
│       ├── metrics.py
│       ├── negative_sampler.py
│       ├── robustness_test.py
│       ├── sc_tgn.py
│       ├── temporal_graph_builder.py
│       └── trainer_alt.py
│
└── phase4/  # [PLANNED] 제약 기반 최적 재배선
    ├── main_phase4.py                # Phase 4 메인 실행 파일 (예정)
    ├── evaluate_rewiring.py          # 재배선 평가 (예정)
    └── src/
        ├── rewiring_optimizer.py     # 재배선 최적화 알고리즘 (예정)
        ├── buffer_calculator.py      # 충격완충력 계산 (예정)
        ├── penalty_calculator.py     # 재고/용량 패널티 계산 (예정)
        ├── constraint_checker.py     # 제약 조건 검증 (예정)
        └── benchmarks.py             # Greedy, Random 벤치마크 (예정)
```

## 📊 File Count Summary

| Directory | Python Files | Purpose |
|-----------|--------------|---------|
| Root | 1 | Utilities |
| phase1/ | 2 | Main execution files |
| phase1/src/ | 4 | Core modules |
| phase2/ | 2 | Main execution files |
| phase2/src/ | 5 | Core modules |
| phase3/ | 6 | Main execution files |
| phase3/src/ | 11 | Core modules |
| phase4/ (planned) | 2 | Main execution files (예정) |
| phase4/src/ (planned) | 5 | Core modules (예정) |
| **Total (Current)** | **31** | **Project files** |
| **Total (Planned)** | **38** | **Including Phase 4** |

## 🔍 File Purpose Quick Reference

### Root Level
- `clear_cache.py` - Cache cleanup utility

### Phase 1 (Production Function Estimation)
**Main Files:**
- `main_phase1.py` - Execute Phase 1 pipeline
- `generate_dummy_data.py` - Generate test data

**Core Modules (src/):**
- `b_matrix_generator.py` - Generate B matrix (firm-sector transaction shares)
- `inventory_module.py` - Zero-shot inventory estimation
- `check_recipe.py` - Validate production recipes
- `debug_deep_dive.py` - Debugging utilities

### Phase 2 (Static Graph Embedding)
**Main Files:**
- `main_phase2.py` - Execute Phase 2 pipeline
- `test_phase2.py` - Phase 2 testing

**Core Modules (src/):**
- `graph_builder.py` - Build static graph with index alignment
- `GraphSAGE.py` - GraphSAGE model (2-layer SAGEConv)
- `sampler.py` - Negative sampling (Historical Hard + Random)
- `loss.py` - TIS-based Risk-Aware BCE Loss
- `trainer.py` - Training loop manager

### Phase 3 (Link Prediction & Evaluation)
**Main Files:**
- `main.py` - Execute Phase 3 pipeline (latest version)
- `main_old.py` - Previous version for reference
- `evaluate_comprehensive.py` - Comprehensive model evaluation
- `generate_temporal_networks.py` - Generate temporal graph structures
- `quick_test.py` - Quick functionality tests
- `test.py` - Phase 3 testing

**Core Modules (src/):**
- `graphseal.py` - GraphSEAL with DGCNN for link prediction
- `sc_tgn.py` - Temporal Graph Network implementation
- `temporal_graph_builder.py` - Build temporal graph events
- `link_predictor.py` - Link prediction interface
- `benchmarks.py` - Heuristic benchmarks (CN, AA, PA)
- `loss.py` - Loss functions for Phase 3
- `trainer_alt.py` - Alternative training approach
- `hybrid_trainer.py` - Hybrid model training
- `metrics.py` - Evaluation metrics (ROC-AUC, Precision@K)
- `negative_sampler.py` - Negative sampling strategies
- `robustness_test.py` - Model robustness testing

### Phase 4 (Constrained Rewiring) [PLANNED]
**Main Files (예정):**
- `main_phase4.py` - Execute Phase 4 pipeline
- `evaluate_rewiring.py` - Evaluate rewiring strategies

**Core Modules (src/) (예정):**
- `rewiring_optimizer.py` - TIS-optimized rewiring algorithm
  * Buffer calculation: f(z_v) × 1/(TIS_v + ε)
  * Final scoring: P(u,v) × Buffer(v) - Penalty_inv
- `buffer_calculator.py` - Compute shock absorption capacity
- `penalty_calculator.py` - Recipe/capacity mismatch penalties
- `constraint_checker.py` - Validate constraints (capacity, recipe)
- `benchmarks.py` - Greedy and Random baselines

## 🎯 Execution Order

```
1. phase1/main_phase1.py
   ↓
   Generates: data/processed/disentangled_recipes.pkl
   
2. phase2/main_phase2.py
   ↓
   Generates: 
   - data/processed/node_embeddings_static.pt
   - data/processed/train_edges.npy
   - data/processed/test_edges.npy
   
3. phase3/main.py
   ↓
   Generates: 
   - Trained link prediction model
   - link_predictions.npy (링크 예측 확률)
   Evaluates: ROC-AUC, Precision@K
   
4. phase4/main_phase4.py [PLANNED]
   ↓
   Input: Phase 3 link predictions + TIS + recipes + financial data
   Processing:
   - Buffer calculation: f(z_v) × 1/(TIS_v + ε)
   - Final scoring: P(u,v) × Buffer(v) - Penalty_inv
   - Constrained rewiring selection
   Generates:
   - rewiring_map.pkl (재배선 매핑)
   - H_prime_rewired.npz (재배선된 네트워크)
   - buffer_scores.npy (충격완충력)
   
5. phase5/main_phase5.py [PLANNED]
   ↓
   Input: H_original + H_prime_rewired
   Processing: Shock propagation simulation
   Evaluates: Economic loss (원본 vs 재배선)
```

## 📝 Configuration Files

Each phase has its own:
- `README.md` - Phase documentation
- `STRUCTURE.txt` - Phase structure details
- `requirements.txt` - Python dependencies
- Quick test scripts (`.sh` files)

## 🔗 Key Dependencies

The phases are interconnected:
- **Phase 2** depends on **Phase 1** output (recipes)
- **Phase 3** depends on **Phase 2** output (embeddings, edges)
- **Phase 4** depends on **Phase 3** output (link predictions) + **Phase 2** (TIS) + **Phase 1** (recipes)
- **Phase 5** depends on **Phase 4** output (rewired network)

All phases share the same raw data in `data/raw/` but generate separate outputs in `data/processed/`.

### Phase 4 추가 입력 요구사항
- **재무 데이터:** 매출, 자산, 영업이익 (기초 체력 계산용)
- **단절 시나리오:** 관세 타격 대상 노드 리스트
- **제약 조건:** 용량 제한, 레시피 임계값

---

**Last Updated:** 2026-01-19  
**Note:** This tree excludes virtual environment files (.venv/) and Python cache (__pycache__/).  
**Phase 4 Status:** 설계 완료, 구현 대기 중
