# GNN Project Structure Summary

**Last Updated:** 2024-01-19  
**Total Python Files:** 31 project files (excluding .venv and packages)

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
└── phase3/  [LINK PREDICTION & EVALUATION]
    ├── README.md                       # Phase 3 documentation
    ├── STRUCTURE.txt                   # Phase 3 structure
    ├── FINAL_SUMMARY.txt               # Phase 3 final summary
    ├── requirements.txt                # Python dependencies
    ├── main.py                         # Main execution file (latest)
    ├── main_old.py                     # Previous version
    ├── quick_test.py                   # Quick test
    ├── test.py                         # Test script
    ├── evaluate_comprehensive.py       # Comprehensive evaluation
    ├── generate_temporal_networks.py   # Temporal network generation
    └── src/
        ├── temporal_graph_builder.py   # TGN temporal data builder
        ├── graphseal.py                # GraphSEAL (DGCNN link prediction)
        ├── sc_tgn.py                   # Temporal Graph Network
        ├── link_predictor.py           # Link predictor
        ├── loss.py                     # Loss functions
        ├── trainer_alt.py              # Alternative trainer
        ├── hybrid_trainer.py           # Hybrid trainer
        ├── benchmarks.py               # Heuristic benchmarks (CN, AA, PA)
        ├── metrics.py                  # Evaluation metrics
        ├── negative_sampler.py         # Negative sampler
        └── robustness_test.py          # Robustness testing
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

### Phase 2: Static Graph Embedding
**Location:** `phase2/main_phase2.py`

**Input:**
- Phase 1 output
- Revenue/Export/Asset/TIS data
- H matrix

**Processing:**
- `graph_builder.py`: Build static graph with index alignment
- Feature generation: Financial + Coordinates + TIS + Industry + Recipe
- Train/Test edge split
- GraphSAGE training
- `sampler.py`: Negative sampling
- `loss.py`: TIS-based loss function

**Output:**
- `node_embeddings_static.pt` (32-dim node embeddings)
- `train_edges.npy` (training edges)
- `test_edges.npy` (test edges)

### Phase 3: Link Prediction & Evaluation
**Location:** `phase3/main.py`

**Input:**
- Phase 2 embeddings
- Edge data

**Processing:**
- GraphSEAL (DGCNN): Subgraph structure learning
- Temporal graph analysis (`temporal_graph_builder.py`)
- Benchmarks: Common Neighbors, Adamic-Adar, etc.

**Output:**
- Trained link prediction model

**Evaluation:**
- ROC-AUC
- Precision@K

## 📊 Data Statistics

- **Nodes (Firms):** 438,946
- **Edges (Transactions):** Millions
- **Time Period:** 2020-2024 (5 years)
- **Embedding Dimension:** 32
- **Recipe Dimension:** 33 (IO table sectors)

## 🔑 Key Components

### Phase 1 Modules
- **BMatrixGenerator:** Generate B matrix from transaction shares
- **ZeroShotInventoryModule:** Estimate production functions

### Phase 2 Modules
- **graph_builder.py:** Static graph construction with alignment
- **GraphSAGE.py:** 2-layer SAGEConv for embeddings
- **sampler.py:** Historical Hard + Random negative sampling
- **loss.py:** TIS-based Risk-Aware BCE Loss

### Phase 3 Modules
- **graphseal.py:** GraphSEAL with DGCNN for link prediction
- **temporal_graph_builder.py:** Build temporal graph events for TGN
- **benchmarks.py:** Heuristic benchmarks (CN, AA, PA)

## 📝 Key Files

| File | Purpose |
|------|---------|
| `clear_cache.py` | Cache management utility |
| `CACHE_GUIDE.md` | Cache system guide |
| `COLUMN_NAME_UPDATE.md` | Column naming guide |
| `columns` | Data specification document |
| `structure` | Project structure document |

## 🎯 Core Algorithms

1. **Production Function Estimation:** Zero-shot learning based
2. **Graph Embedding:** GraphSAGE (Inductive learning)
3. **Link Prediction:** GraphSEAL (DGCNN) + Temporal analysis
4. **Negative Sampling:** Historical Hard + Random
5. **Loss Function:** TIS-based Risk-Aware BCE Loss

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

**Note:** This structure reflects the current state of the project. The `structure` file contains the full detailed documentation.
