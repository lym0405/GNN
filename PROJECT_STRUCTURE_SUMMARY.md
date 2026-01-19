# GNN Project Structure Summary

**Last Updated:** 2025-01-19  
**Total Python Files:** 43 project files (excluding .venv and packages)  
**Pipeline Status:** Phases 1-5 fully implemented and integrated

## 📁 Directory Structure

```
GNN/
├── clear_cache.py                      # Cache cleanup utility
│
├── data/
│   ├── raw/                            # Original datasets (read-only)
│   │   ├── posco_network_capital_consumergoods_removed_{year}.csv
│   │   │                               # Transaction networks (year = 2020-2023)
│   │   ├── H_csr_model2.npz            # Sparse adjacency matrix (438K×438K)
│   │   ├── firm_to_idx_model2.csv      # Firm ID to index mapping
│   │   ├── vat_*_company_list_*.csv    # Company information
│   │   ├── A_33.csv                    # National IO table (33×33)
│   │   ├── tg_2024_filtered.csv        # Revenue data
│   │   ├── export_estimation_value_final.csv # Export values
│   │   ├── asset_final_2024_6차.csv     # Asset estimation
│   │   └── shock_after_P_v2.csv        # TIS risk scores
│   │
│   └── processed/                      # Generated outputs
│       ├── disentangled_recipes.pkl    # Phase 1: Production functions (33-dim)
│       ├── recipes_dataframe.csv       # Phase 1: Recipe dataframe
│       ├── recipe_validation_report.csv # Phase 1: Validation report
│       ├── B_matrix.npy                # Phase 1: B matrix
│       ├── X_feature_matrix.npy        # Phase 2: Feature matrix
│       ├── recipe_features_cache.npy   # Phase 2: Recipe cache
│       ├── tis_score_normalized.npy    # Phase 2: Normalized TIS scores
│       ├── node_embeddings_static.pt   # Phase 2: GraphSAGE embeddings (32-dim)
│       ├── train_edges.npy             # Phase 2: Training edges (80%)
│       └── test_edges.npy              # Phase 2: Test edges (20%)
│
├── results/                            # Execution results
│   └── quick_test/                     # Quick test outputs
│
├── trash/                              # Temporary files (empty)
│
├── phase1/  [PRODUCTION FUNCTION ESTIMATION]
│   ├── README.md                       # Phase 1 documentation
│   ├── STRUCTURE.txt                   # Phase 1 structure
│   ├── requirements.txt                # Python dependencies
│   ├── quick_test.sh                   # Quick test script
│   ├── generate_dummy_data.py          # Dummy data generator
│   ├── main_phase1.py                  # Main execution file
│   └── src/
│       ├── b_matrix_generator.py       # BMatrixGenerator
│       ├── inventory_module.py         # ZeroShotInventoryModule
│       ├── check_recipe.py             # Recipe validation
│       └── debug_deep_dive.py          # Debug utilities
│
├── phase2/  [STATIC GRAPH EMBEDDING]
│   ├── README.md                       # Phase 2 documentation
│   ├── STRUCTURE.txt                   # Phase 2 structure
│   ├── requirements.txt                # Python dependencies
│   ├── quick_test_phase2.sh            # Quick test script
│   ├── main_phase2.py                  # Main execution file
│   ├── test_phase2.py                  # Test script
│   └── src/
│       ├── graph_builder.py            # Static graph builder
│       ├── GraphSAGE.py                # GraphSAGE model (2-layer SAGEConv)
│       ├── sampler.py                  # Negative sampling
│       ├── loss.py                     # RiskAwareBCELoss
│       └── trainer.py                  # Training loop manager
│
│
├── phase3/  [LINK PREDICTION & EVALUATION]
│   ├── README.md                       # Phase 3 documentation
│   ├── STRUCTURE.txt                   # Phase 3 structure
│   ├── FINAL_SUMMARY.txt               # Phase 3 final summary
│   ├── PHASE3_HISTORICAL_NEGATIVES_FIX.md # Historical negatives fix doc
│   ├── requirements.txt                # Python dependencies
│   ├── main.py                         # Main execution file (latest)
│   ├── main_old.py                     # Previous version
│   ├── quick_test.py                   # Quick test
│   ├── test.py                         # Test script
│   ├── test_historical_negatives.py    # Test historical negative sampling
│   ├── evaluate_comprehensive.py       # Comprehensive evaluation
│   ├── generate_temporal_networks.py   # Temporal network generation
│   └── src/
│       ├── temporal_graph_builder.py   # TGN temporal data builder
│       ├── graphseal.py                # GraphSEAL (DGCNN link prediction)
│       ├── sc_tgn.py                   # Temporal Graph Network
│       ├── link_predictor.py           # Link predictor
│       ├── loss.py                     # Loss functions
│       ├── trainer_alt.py              # Alternative trainer
│       ├── hybrid_trainer.py           # Hybrid trainer
│       ├── benchmarks.py               # Heuristic benchmarks (CN, AA, PA)
│       ├── metrics.py                  # Evaluation metrics
│       ├── negative_sampler.py         # Negative sampler (FIXED)
│       └── robustness_test.py          # Robustness testing
│
├── phase4/  [CONSTRAINED REWIRING]
│   ├── README.md                       # Phase 4 documentation
│   ├── PHASE4_DESIGN.md                # Design document
│   ├── PHASE4_SUMMARY.md               # Summary and results
│   ├── requirements.txt                # Python dependencies
│   ├── main_phase4.py                  # Main execution file
│   ├── test_phase4.py                  # Test script
│   ├── config/                         # Configuration files
│   └── src/
│       ├── rewiring_optimizer.py       # Rewiring optimization algorithm
│       ├── buffer_calculator.py        # Shock buffer calculation
│       ├── penalty_calculator.py       # Inventory/capacity penalties
│       ├── constraint_checker.py       # Constraint validation
│       ├── benchmarks.py               # Greedy, Random benchmarks
│       └── evaluate_rewiring.py        # Rewiring evaluation
│
└── phase5/  [HISTORICAL VALIDATION]
    ├── README.md                       # Phase 5 documentation
    ├── PHASE5_DESIGN.md                # Design document
    ├── PHASE5_IMPLEMENTATION.md        # Implementation details
    ├── main_phase5.py                  # Main execution file
    └── src/
        ├── shock_injector.py           # Shock injection (2019 Japan)
        ├── ksic_matcher.py             # KSIC code matching
        └── evaluator.py                # Historical validation metrics
```

## 🔄 Execution Flow

### Phase 1: Production Function Estimation
**Location:** `phase1/main_phase1.py`

**Input:**
- IO table (A_33.csv)
- Transaction network (H_csr_model2.npz)
- Company information
- Revenue data

**Processing:**
- BMatrixGenerator: Generate B matrix (firm-sector transaction shares)
- ZeroShotInventoryModule: Estimate production functions

**Output:**
- `disentangled_recipes.pkl` (33-dimensional production functions per firm)
- `recipes_dataframe.csv` (human-readable format)
- `recipe_validation_report.csv` (validation metrics)

### Phase 2: Static Graph Embedding
**Location:** `phase2/main_phase2.py`

**Input:**
- Phase 1 output (recipes)
- Revenue/Export/Asset/TIS data
- H matrix (transaction network)

**Processing:**
- `graph_builder.py`: Build static graph with index alignment
- Feature generation: Financial + Coordinates + TIS + Industry + Recipe (73-dim)
- Train/Test edge split (80/20)
- GraphSAGE training (2-layer SAGEConv)
- `sampler.py`: Historical Hard (50%) + Random (50%) negative sampling
- `loss.py`: TIS-based Risk-Aware BCE Loss

**Output:**
- `node_embeddings_static.pt` (32-dim node embeddings)
- `X_feature_matrix.npy` (73-dim feature matrix)
- `train_edges.npy`, `test_edges.npy`
- `recipe_features_cache.npy`
- `tis_score_normalized.npy`

### Phase 3: Link Prediction & Evaluation
**Location:** `phase3/main.py`

**Input:**
- Phase 2 embeddings and features
- Temporal edge data (2020-2024)
- Historical negative edges (14,550 edges from 4 years)

**Processing:**
- `temporal_graph_builder.py`: Build temporal snapshots
- GraphSEAL (DGCNN): Subgraph structure learning for link prediction
- SC-TGN: Temporal graph network with memory
- `negative_sampler.py`: Load historical negatives + generate new negatives
- Benchmarks: Common Neighbors, Adamic-Adar, Preferential Attachment

**Output:**
- Trained link prediction model
- Evaluation metrics (ROC-AUC, Precision@K)
- Temporal evolution analysis

**Key Fix (Jan 2025):**
- Fixed historical negative sampling (was loading 0, now loads 14,550 edges)
- Corrected Korean column name matching in `firm_to_idx_model2.csv`

### Phase 4: Constrained Rewiring
**Location:** `phase4/main_phase4.py`

**Input:**
- Phase 3 predictions (top-K candidate links)
- Production functions (Phase 1)
- Network structure and constraints

**Processing:**
- `rewiring_optimizer.py`: Optimize rewiring under constraints
- `buffer_calculator.py`: Calculate shock absorption capacity
- `penalty_calculator.py`: Compute inventory and capacity penalties
- `constraint_checker.py`: Validate feasibility
- `benchmarks.py`: Compare with Greedy and Random strategies

**Output:**
- Optimized rewiring recommendations
- Constraint satisfaction report
- Performance comparison with baselines

**Constraints:**
- Inventory capacity limits
- Production capacity limits
- Geographic distance constraints
- Recipe compatibility

### Phase 5: Historical Validation
**Location:** `phase5/main_phase5.py`

**Input:**
- Full pipeline outputs (Phases 1-4)
- Historical event data (2019 Japan Export Restriction)

**Processing:**
- `shock_injector.py`: Inject historical shock to network
- `ksic_matcher.py`: Match affected industries (semiconductors, displays)
- `evaluator.py`: Compare predictions vs. actual outcomes

**Output:**
- Validation metrics (precision, recall, accuracy)
- Comparison: Model predictions vs. actual network evolution
- Case study analysis

**Historical Event:**
- Event: July 2019 Japan export restrictions on South Korea
- Affected: Semiconductors (C261), Display panels (C262)
- Impact: Supply chain disruptions, forced rewiring

## 📊 Data Statistics

- **Nodes (Firms):** 438,946
- **Edges (Transactions):** Millions (varies by year)
- **Time Period:** 2020-2024 (5 years)
- **Node Embedding Dimension:** 32 (GraphSAGE output)
- **Node Feature Dimension:** 73 (input features)
- **Recipe Dimension:** 33 (IO table sectors)
- **Historical Negative Edges:** 14,550 (across 4 years: 2020-2023)
- **IO Sectors:** 33 (Korean Input-Output table classification)

## 🔑 Key Components

### Phase 1 Modules
- **BMatrixGenerator:** Generate B matrix from transaction shares
  - Maps firms to 33 IO sectors using `IO상품_단일_대분류_코드`
- **ZeroShotInventoryModule:** Estimate production functions
  - 33-dimensional recipe vectors per firm

### Phase 2 Modules
- **graph_builder.py:** Static graph construction with alignment
  - 73-dim features: Financial (3) + Coordinates (2) + TIS (1) + Industry (33) + Recipe (33) + Other (1)
- **GraphSAGE.py:** 2-layer SAGEConv for embeddings
  - Input: 73-dim → Hidden: 64-dim → Output: 32-dim
- **sampler.py:** Historical Hard (50%) + Random (50%) negative sampling
  - Optimized from 1:9 to 1:2 negative ratio
  - Batch size increased from 1024 to 4096
- **loss.py:** TIS-based Risk-Aware BCE Loss
  - Weighted by supply chain risk scores

### Phase 3 Modules
- **graphseal.py:** GraphSEAL with DGCNN for link prediction
  - Subgraph extraction and structure learning
- **temporal_graph_builder.py:** Build temporal graph events for TGN
  - Snapshots across 2020-2024
- **negative_sampler.py:** Load historical negatives + generate new negatives
  - **FIXED:** Korean column name matching (`사업자등록번호`)
  - Now loads 14,550 historical edges (was 0)
- **benchmarks.py:** Heuristic benchmarks
  - Common Neighbors (CN), Adamic-Adar (AA), Preferential Attachment (PA)

### Phase 4 Modules
- **rewiring_optimizer.py:** Constrained optimization for rewiring
  - Linear programming or heuristic search
- **buffer_calculator.py:** Shock absorption capacity
  - Based on production functions and inventory
- **penalty_calculator.py:** Inventory and capacity penalties
  - Soft constraints for realistic rewiring
- **constraint_checker.py:** Hard constraint validation
  - Recipe compatibility, capacity limits, distance

### Phase 5 Modules
- **shock_injector.py:** Historical shock injection
  - 2019 Japan export restrictions
  - Removes/weakens edges from Japanese suppliers
- **ksic_matcher.py:** Industry code matching
  - Maps KSIC codes to affected sectors (C261, C262)
- **evaluator.py:** Validation metrics
  - Compares model predictions with actual 2019-2020 network changes

## 📝 Key Files

| File | Purpose |
|------|---------|
| `clear_cache.py` | Cache management utility |
| `CACHE_GUIDE.md` | Cache system guide |
| `COLUMN_NAME_UPDATE.md` | Column naming guide |
| `columns` | Data specification document |
| `structure` | Project structure document |

## 🎯 Core Algorithms

1. **Production Function Estimation (Phase 1):** 
   - Zero-shot learning based on IO table and transaction shares
   - 33-dimensional recipe vectors per firm

2. **Graph Embedding (Phase 2):** 
   - GraphSAGE (Inductive learning)
   - 2-layer SAGEConv: 73-dim → 64-dim → 32-dim

3. **Link Prediction (Phase 3):** 
   - GraphSEAL (DGCNN) for subgraph structure
   - SC-TGN for temporal evolution
   - Historical + Random negative sampling (50/50)

4. **Rewiring Optimization (Phase 4):**
   - Constrained optimization with hard/soft constraints
   - Buffer calculation and penalty functions
   - Greedy and Random baselines for comparison

5. **Historical Validation (Phase 5):**
   - Shock injection (2019 Japan export restrictions)
   - KSIC matching for affected industries
   - Precision/Recall evaluation vs. actual outcomes

## 🐛 Recent Bug Fixes & Optimizations

### Phase 3: Historical Negatives Fix (Jan 2025) - CRITICAL
- **Problem:** Historical negatives always showed 0
- **Cause:** Korean column name mismatch in `firm_to_idx_model2.csv`
- **Solution:** Fixed column priority to check `사업자등록번호` first
- **Impact:** Now loads 14,550 historical edges across 4 years
- **Commit:** `3f4dde0`

### Phase 2: Training Optimization (Dec 2024)
- **Changes:** 
  - Negative sampling ratio: 1:9 → 1:2
  - Batch size: 1024 → 4096
- **Impact:** ~3-4x faster training

### Cache System
- Automatic caching in `data/processed/cache/`
- Use `clear_cache.py` to force rebuild
- See `CACHE_GUIDE.md` for details

## 📦 Data Files

### Raw Data (data/raw/)
- Transaction networks: `posco_network_capital_consumergoods_removed_{year}.csv` (2020-2023)
- Sparse adjacency matrix (438K×438K)
- Company information with coordinates
- National IO table (33×33)
- Revenue (`tg_2024_filtered.csv`), Export, Asset, TIS data

### Processed Data (data/processed/)
- Production functions (recipes)
- Feature matrices
- Node embeddings
- Train/Test edges
- Cached computations

---

**Note:** This structure reflects the current state of the project (January 2025). All 5 phases are implemented and integrated. For detailed column specifications, see `COLUMN_NAME_UPDATE.md`. For cache management, see `CACHE_GUIDE.md`. For full project documentation, see `structure` file and phase-specific README files.
