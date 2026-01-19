# GNN Project - Python Files Tree

**Last Updated:** 2025-01-19  
**Total Files:** 43 Python files (Phases 1-5 fully implemented)

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
│   ├── test_historical_negatives.py      # NEW: Test historical negative loading
│   └── src/
│       ├── benchmarks.py
│       ├── graphseal.py
│       ├── hybrid_trainer.py
│       ├── link_predictor.py
│       ├── loss.py
│       ├── metrics.py
│       ├── negative_sampler.py           # FIXED: Korean column name matching
│       ├── robustness_test.py
│       ├── sc_tgn.py
│       ├── temporal_graph_builder.py
│       └── trainer_alt.py
│
├── phase4/  # IMPLEMENTED
│   ├── main_phase4.py
│   ├── test_phase4.py
│   └── src/
│       ├── benchmarks.py
│       ├── buffer_calculator.py
│       ├── constraint_checker.py
│       ├── evaluate_rewiring.py
│       ├── penalty_calculator.py
│       └── rewiring_optimizer.py
│
└── phase5/  # IMPLEMENTED
    ├── main_phase5.py
    └── src/
        ├── evaluator.py
        ├── ksic_matcher.py
        └── shock_injector.py
```

## 📊 File Count Summary

| Directory | Python Files | Status |
|-----------|--------------|--------|
| Root | 1 | ✅ Complete |
| phase1/ | 2 | ✅ Complete |
| phase1/src/ | 4 | ✅ Complete |
| phase2/ | 2 | ✅ Complete |
| phase2/src/ | 5 | ✅ Complete |
| phase3/ | 7 | ✅ Complete (Bug Fixed) |
| phase3/src/ | 11 | ✅ Complete (Bug Fixed) |
| phase4/ | 2 | ✅ Complete |
| phase4/src/ | 6 | ✅ Complete |
| phase5/ | 1 | ✅ Complete |
| phase5/src/ | 3 | ✅ Complete |
| **Total** | **43** | **All Phases Implemented** |

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
- `generate_temporal_networks.py` - Generate temporal snapshots
- `quick_test.py` - Quick test script
- `test.py` - Full testing script
- `test_historical_negatives.py` - **NEW:** Test historical negative loading (Jan 2025)

**Core Modules (src/):**
- `temporal_graph_builder.py` - Build temporal graph data for TGN
- `graphseal.py` - GraphSEAL (DGCNN-based link prediction)
- `sc_tgn.py` - Supply Chain Temporal Graph Network
- `link_predictor.py` - Link prediction interface
- `loss.py` - Loss functions for link prediction
- `trainer_alt.py` - Alternative training loop
- `hybrid_trainer.py` - Hybrid training approach
- `benchmarks.py` - Heuristic baselines (CN, AA, PA)
- `metrics.py` - Evaluation metrics
- `negative_sampler.py` - **FIXED:** Historical + Random negative sampling (Korean column names)
- `robustness_test.py` - Model robustness testing

### Phase 4 (Constrained Rewiring) - **✅ IMPLEMENTED**
**Main Files:**
- `main_phase4.py` - Execute Phase 4 rewiring optimization
- `test_phase4.py` - Phase 4 testing

**Core Modules (src/):**
- `rewiring_optimizer.py` - Constrained rewiring optimization algorithm
- `buffer_calculator.py` - Calculate shock absorption capacity
- `penalty_calculator.py` - Inventory and capacity penalty functions
- `constraint_checker.py` - Hard constraint validation
- `benchmarks.py` - Greedy and Random baseline strategies
- `evaluate_rewiring.py` - Evaluate rewiring quality

### Phase 5 (Historical Validation) - **✅ IMPLEMENTED**
**Main Files:**
- `main_phase5.py` - Execute Phase 5 historical validation

**Core Modules (src/):**
- `shock_injector.py` - Inject historical shock (2019 Japan export restrictions)
- `ksic_matcher.py` - Match KSIC codes to affected industries
- `evaluator.py` - Evaluate model predictions vs. actual outcomes
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
   - data/processed/node_embeddings_static.pt (32-dim embeddings)
   - data/processed/X_feature_matrix.npy (73-dim features)
   - data/processed/train_edges.npy (80% split)
   - data/processed/test_edges.npy (20% split)
   - data/processed/recipe_features_cache.npy
   - data/processed/tis_score_normalized.npy
   
3. phase3/main.py
   ↓
   Input: 
   - Phase 2 embeddings and features
   - Historical negatives (14,550 edges from 2020-2023)
   Processing:
   - Temporal graph building
   - GraphSEAL + SC-TGN training
   - Historical + Random negative sampling (50/50)
   Generates: 
   - Trained link prediction model
   - link_predictions.npy (link probabilities)
   Evaluates: ROC-AUC, Precision@K
   
4. phase4/main_phase4.py
   ↓
   Input: 
   - Phase 3 link predictions (top-K candidates)
   - TIS scores + production recipes + financial data
   Processing:
   - Buffer calculation: f(z_v) × 1/(TIS_v + ε)
   - Final scoring: P(u,v) × Buffer(v) - Penalty_inv
   - Constrained rewiring selection
   Generates:
   - rewiring_map.pkl (optimal rewiring recommendations)
   - constraint_report.csv
   Evaluates:
   - Buffer improvement vs. Greedy/Random baselines
   
5. phase5/main_phase5.py
   ↓
   Input:
   - Full pipeline outputs (Phases 1-4)
   - Historical data (2019-2020 network evolution)
   Processing:
   - Inject 2019 Japan export shock
   - Match affected industries (C261, C262)
   - Compare predictions vs. actual rewiring
   Generates:
   - validation_results.csv
   Evaluates:
   - Precision, Recall, F1-score
   - Case study: Did model predict actual rewiring patterns?
```
   Evaluates:
   - Precision, Recall, F1-score
   - Case study: Did model predict actual rewiring patterns?
```

## 🐛 Recent Updates

### Phase 3 Critical Fix (Jan 2025)
- **File:** `phase3/src/negative_sampler.py`
- **Issue:** Historical negatives always loaded 0 edges
- **Cause:** Korean column name mismatch (`사업자등록번호` vs. `firm_id`)
- **Fix:** Updated column priority to check Korean names first
- **Result:** Now loads 14,550 historical edges (2020-2023)
- **Test:** Added `test_historical_negatives.py` to verify loading

### Phase 2 Optimization (Dec 2024)
- **File:** `phase2/src/sampler.py`
- **Changes:**
  - Negative sampling ratio: 1:9 → 1:2
  - Batch size: 1024 → 4096
- **Impact:** ~3-4x faster training

---

**Last Verified:** 2025-01-19  
**Pipeline Status:** ✅ All 5 phases fully implemented and tested
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
