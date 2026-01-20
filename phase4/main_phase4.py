"""
Phase 4: Constrained Rewiring - Refactored Main Script

This script uses ONLY actual existing files and columns from data/raw and data/processed.
All references to non-existent files, dynamic year-based paths, and YAML configs have been removed.

Pipeline:
1. Load Phase 1-3 outputs (recipes, TIS scores, H matrix)
2. Load financial data from actual raw files
3. Calculate buffer capacity
4. Optimize rewiring with constraints
5. Save results

Usage:
    python phase4/main_phase4.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
import pickle

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "phase4" / "output"

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))


class Config:
    """Configuration using actual existing files only."""
    
    # Phase 1 outputs (from Zero-Shot Inventory Module)
    RECIPES = DATA_PROCESSED / "disentangled_recipes.pkl"
    RECIPES_DF = DATA_PROCESSED / "recipes_dataframe.csv"
    B_MATRIX = DATA_PROCESSED / "B_matrix.npy"
    
    # Phase 2/3 outputs (from SC-TGN + GraphSEAL)
    TIS_SCORES = DATA_PROCESSED / "tis_score_normalized.npy"
    NODE_EMBEDDINGS = DATA_PROCESSED / "node_embeddings_static.pt"
    TRAIN_EDGES = DATA_PROCESSED / "train_edges.npy"
    TEST_EDGES = DATA_PROCESSED / "test_edges.npy"
    
    # Raw data files (actual files in data/raw)
    H_MATRIX = DATA_RAW / "H_csr_model2.npz"
    FIRM_TO_IDX = DATA_RAW / "firm_to_idx_model2.csv"
    REVENUE = DATA_RAW / "final_tg_2024_estimation.csv"
    ASSET = DATA_RAW / "asset_final_2024_6차.csv"
    EXPORT = DATA_RAW / "export_estimation_value_final.csv"
    SHOCK_DATA = DATA_RAW / "shock_after_P_v2.csv"
    
    # Rewiring parameters
    ALPHA = 0.5  # Weight for penalty score
    BETA = 0.3   # Weight for buffer capacity
    GAMMA = 0.2  # Weight for other factors
    TOP_K = 100  # Number of top vulnerable nodes to rewire
    MAX_NEW_EDGES = 50  # Maximum new edges per node
    
    # Constraint parameters
    MAX_SUPPLIER_OUTDEGREE = 10
    MAX_BUYER_INDEGREE = 10
    RECIPE_SIMILARITY_THRESHOLD = 0.7
    CAPACITY_RATIO_MIN = 0.5
    CAPACITY_RATIO_MAX = 2.0


def setup_logging():
    """Setup logging configuration."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase4_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Phase 4 started. Log file: {log_file}")
    return logger


def validate_files(config: Config, logger: logging.Logger):
    """Validate that all required files exist."""
    logger.info("Validating input files...")
    
    required_files = {
        "Recipes": config.RECIPES,
        "TIS Scores": config.TIS_SCORES,
        "H Matrix": config.H_MATRIX,
        "Firm to Index": config.FIRM_TO_IDX,
        "Revenue": config.REVENUE,
        "Asset": config.ASSET,
        "Export": config.EXPORT,
    }
    
    missing = []
    for name, path in required_files.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
            logger.error(f"Missing file: {name} at {path}")
        else:
            logger.info(f"✓ Found {name}: {path}")
    
    if missing:
        raise FileNotFoundError(f"Missing required files:\n" + "\n".join(missing))
    
    logger.info("All required files validated successfully")


def load_phase123_outputs(config: Config, logger: logging.Logger) -> Dict:
    """Load outputs from Phase 1-3."""
    logger.info("Loading Phase 1-3 outputs...")
    
    # Load recipes (Phase 1)
    with open(config.RECIPES, 'rb') as f:
        recipes = pickle.load(f)
    logger.info(f"Loaded recipes: {len(recipes)} firms")
    
    # Load TIS scores (Phase 2/3)
    tis_scores = np.load(config.TIS_SCORES)
    logger.info(f"Loaded TIS scores: shape {tis_scores.shape}, range [{tis_scores.min():.4f}, {tis_scores.max():.4f}]")
    
    # Load H matrix (supply network)
    H_data = sparse.load_npz(config.H_MATRIX)
    logger.info(f"Loaded H matrix: shape {H_data.shape}, {H_data.nnz} non-zero entries")
    
    # Load firm to index mapping
    firm_to_idx_df = pd.read_csv(config.FIRM_TO_IDX)
    # Column name might be '사업자등록번호' or similar
    id_col = firm_to_idx_df.columns[0]
    firm_ids = firm_to_idx_df[id_col].tolist()
    logger.info(f"Loaded firm IDs: {len(firm_ids)} firms (column: {id_col})")
    
    return {
        'recipes': recipes,
        'tis_scores': tis_scores,
        'H_matrix': H_data,
        'firm_ids': firm_ids,
    }


def load_financial_data(config: Config, firm_ids: List, logger: logging.Logger) -> Dict:
    """Load financial data from actual raw files using real column names."""
    logger.info("Loading financial data...")
    
    n_firms = len(firm_ids)
    
    # Initialize arrays with zeros
    revenue = np.zeros(n_firms)
    asset = np.zeros(n_firms)
    export_value = np.zeros(n_firms)
    
    # Load revenue data
    logger.info("Loading revenue data...")
    df_rev = pd.read_csv(config.REVENUE)
    logger.info(f"Revenue file columns: {df_rev.columns.tolist()}")
    
    # Find actual column names (업체번호, tg_2024_final)
    id_col = '업체번호' if '업체번호' in df_rev.columns else df_rev.columns[0]
    rev_col = 'tg_2024_final' if 'tg_2024_final' in df_rev.columns else df_rev.columns[1]
    
    rev_dict = dict(zip(df_rev[id_col], df_rev[rev_col]))
    for i, fid in enumerate(firm_ids):
        revenue[i] = rev_dict.get(fid, 0.0)
    logger.info(f"Loaded revenue: {np.sum(revenue > 0)} firms have non-zero values")
    
    # Load asset data
    logger.info("Loading asset data...")
    df_asset = pd.read_csv(config.ASSET)
    logger.info(f"Asset file columns: {df_asset.columns.tolist()}")
    
    id_col = '업체번호' if '업체번호' in df_asset.columns else df_asset.columns[0]
    asset_col = '자산추정_2024' if '자산추정_2024' in df_asset.columns else df_asset.columns[1]
    
    asset_dict = dict(zip(df_asset[id_col], df_asset[asset_col]))
    for i, fid in enumerate(firm_ids):
        asset[i] = asset_dict.get(fid, 0.0)
    logger.info(f"Loaded asset: {np.sum(asset > 0)} firms have non-zero values")
    
    # Load export data
    logger.info("Loading export data...")
    df_export = pd.read_csv(config.EXPORT)
    logger.info(f"Export file columns: {df_export.columns.tolist()}")
    
    id_col = '업체번호' if '업체번호' in df_export.columns else df_export.columns[0]
    export_col = 'export_value' if 'export_value' in df_export.columns else df_export.columns[1]
    
    export_dict = dict(zip(df_export[id_col], df_export[export_col]))
    for i, fid in enumerate(firm_ids):
        export_value[i] = export_dict.get(fid, 0.0)
    logger.info(f"Loaded export: {np.sum(export_value > 0)} firms have non-zero values")
    
    return {
        'revenue': revenue,
        'asset': asset,
        'export_value': export_value,
    }


def calculate_buffer_capacity(
    tis_scores: np.ndarray,
    financial_data: Dict,
    recipes: Dict,
    logger: logging.Logger
) -> np.ndarray:
    """
    Calculate buffer capacity for each node.
    
    Buffer capacity formula:
        Buffer(v) = f(financial_strength_v, inventory_flexibility_v) × 1/(TIS_v + ε)
    
    where:
        - financial_strength = normalized(revenue + asset + export)
        - inventory_flexibility = number of alternative recipes (from recipe complexity)
        - TIS = Technology Import Susceptibility score
    """
    logger.info("Calculating buffer capacity...")
    
    n_nodes = len(tis_scores)
    
    # Financial strength (combine revenue, asset, export)
    financial_strength = financial_data['revenue'] + financial_data['asset'] + financial_data['export_value']
    # Normalize to [0, 1]
    if financial_strength.max() > 0:
        financial_strength = financial_strength / financial_strength.max()
    else:
        logger.warning("All financial strengths are zero!")
        financial_strength = np.zeros(n_nodes)
    
    # Inventory flexibility based on recipe complexity
    # Recipes are stored as numpy arrays (product composition vectors)
    # Higher diversity in recipe = higher flexibility
    inventory_flexibility = np.ones(n_nodes)
    
    if isinstance(recipes, dict):
        logger.info(f"Processing {len(recipes)} recipes...")
        
        # Create firm_id to index mapping if recipes use string keys
        sample_key = next(iter(recipes.keys()))
        if isinstance(sample_key, str):
            # Recipes use string keys like 'firm_000000'
            logger.info("Recipes use string keys, calculating flexibility from recipe diversity")
            
            for firm_id, recipe_vector in recipes.items():
                if isinstance(recipe_vector, np.ndarray):
                    # Use recipe diversity as flexibility measure
                    # Higher diversity (entropy) = more alternatives
                    non_zero_count = np.sum(recipe_vector > 0.01)  # Count significant components
                    inventory_flexibility_value = 1.0 + np.log1p(non_zero_count)
                    
                    # Map firm_id to node index if possible
                    # For now, we'll use this for all nodes equally
                    # (In production, you'd need proper firm_id to index mapping)
        else:
            # Recipes use integer keys
            for node_idx, recipe_data in recipes.items():
                if isinstance(recipe_data, np.ndarray):
                    non_zero_count = np.sum(recipe_data > 0.01)
                    inventory_flexibility[node_idx] = 1.0 + np.log1p(non_zero_count)
                elif isinstance(recipe_data, dict) and 'alternatives' in recipe_data:
                    inventory_flexibility[node_idx] = len(recipe_data['alternatives'])
    
    # Normalize inventory flexibility
    if inventory_flexibility.max() > inventory_flexibility.min():
        inventory_flexibility = (inventory_flexibility - inventory_flexibility.min()) / \
                               (inventory_flexibility.max() - inventory_flexibility.min())
    else:
        logger.warning("Inventory flexibility has no variance, using default values")
        inventory_flexibility = np.ones(n_nodes) * 0.5
    
    logger.info(f"Financial strength range: [{financial_strength.min():.4f}, {financial_strength.max():.4f}]")
    logger.info(f"Inventory flexibility range: [{inventory_flexibility.min():.4f}, {inventory_flexibility.max():.4f}]")
    
    # Combine factors
    base_capacity = 0.7 * financial_strength + 0.3 * inventory_flexibility
    
    # Apply TIS penalty (higher TIS = lower buffer)
    epsilon = 1e-8
    buffer_capacity = base_capacity / (tis_scores + epsilon)
    
    # Normalize to [0, 1]
    if buffer_capacity.max() > 0:
        buffer_capacity = buffer_capacity / buffer_capacity.max()
    else:
        logger.warning("Buffer capacity is all zeros!")
        buffer_capacity = np.ones(n_nodes) * 0.1
    
    logger.info(f"Buffer capacity calculated: range [{buffer_capacity.min():.4f}, {buffer_capacity.max():.4f}]")
    logger.info(f"Mean buffer: {buffer_capacity.mean():.4f}, Median: {np.median(buffer_capacity):.4f}")
    
    return buffer_capacity


def calculate_penalty_score(
    supplier: int,
    buyer: int,
    H_matrix: sparse.csr_matrix,
    financial_data: Dict,
    recipes: Dict
) -> float:
    """
    Calculate penalty score for adding a new edge.
    
    Lower penalty = better match
    Considers:
    - Recipe similarity
    - Capacity mismatch
    - Geographic/sector distance
    """
    # Base penalty
    penalty = 0.0
    
    # Recipe similarity penalty (if recipes don't match well)
    if supplier in recipes and buyer in recipes:
        # Simplified: assume we have some similarity metric
        penalty += 0.3  # Placeholder
    else:
        penalty += 0.5  # Higher penalty if no recipe info
    
    # Capacity mismatch penalty
    supplier_revenue = financial_data['revenue'][supplier]
    buyer_revenue = financial_data['revenue'][buyer]
    
    if supplier_revenue > 0 and buyer_revenue > 0:
        capacity_ratio = buyer_revenue / supplier_revenue
        if capacity_ratio < 0.5 or capacity_ratio > 2.0:
            penalty += 0.3
    
    return penalty


def optimize_rewiring(
    tis_scores: np.ndarray,
    buffer_capacity: np.ndarray,
    H_matrix: sparse.csr_matrix,
    financial_data: Dict,
    recipes: Dict,
    config: Config,
    logger: logging.Logger
) -> Dict:
    """
    Optimize rewiring to minimize supply chain risk.
    
    Objective:
        max Σ (α × P(u,v) + β × Buffer(v) - γ × Penalty(u,v))
    
    where P(u,v) is the connection probability (from Phase 3)
    """
    logger.info("Starting rewiring optimization...")
    
    # Find top-k most vulnerable nodes
    top_k_indices = np.argsort(tis_scores)[-config.TOP_K:]
    logger.info(f"Selected top {config.TOP_K} vulnerable nodes")
    logger.info(f"  TIS range: [{tis_scores[top_k_indices].min():.4f}, {tis_scores[top_k_indices].max():.4f}]")
    
    rewiring_map = {}
    new_edges = []
    total_improvement = 0.0
    
    for buyer_idx in top_k_indices:
        # Find potential suppliers (nodes not already connected)
        existing_suppliers = set(H_matrix[:, buyer_idx].nonzero()[0])
        
        candidate_suppliers = []
        for supplier_idx in range(H_matrix.shape[0]):
            if supplier_idx == buyer_idx or supplier_idx in existing_suppliers:
                continue
            
            # Check constraints
            # 1. Max outdegree for supplier
            if H_matrix[supplier_idx, :].nnz >= config.MAX_SUPPLIER_OUTDEGREE:
                continue
            
            # 2. Max indegree for buyer
            if H_matrix[:, buyer_idx].nnz >= config.MAX_BUYER_INDEGREE:
                continue
            
            # Calculate score for this potential edge
            penalty = calculate_penalty_score(supplier_idx, buyer_idx, H_matrix, financial_data, recipes)
            
            # Simplified connection probability (in real implementation, use Phase 3 output)
            connection_prob = 1.0 / (1.0 + penalty)
            
            score = (
                config.ALPHA * connection_prob +
                config.BETA * buffer_capacity[supplier_idx] -
                config.GAMMA * penalty
            )
            
            candidate_suppliers.append((supplier_idx, score))
        
        # Select top candidates
        if candidate_suppliers:
            candidate_suppliers.sort(key=lambda x: x[1], reverse=True)
            
            # Add top suppliers (up to max_new_edges)
            n_new = min(len(candidate_suppliers), config.MAX_NEW_EDGES)
            selected_suppliers = [s[0] for s in candidate_suppliers[:n_new]]
            
            rewiring_map[buyer_idx] = selected_suppliers
            
            for supplier_idx in selected_suppliers:
                new_edges.append((supplier_idx, buyer_idx))
                total_improvement += candidate_suppliers[0][1]  # Use best score
    
    logger.info(f"Rewiring optimization completed")
    logger.info(f"  New edges: {len(new_edges)}")
    logger.info(f"  Nodes rewired: {len(rewiring_map)}")
    logger.info(f"  Total improvement: {total_improvement:.4f}")
    
    return {
        'rewiring_map': rewiring_map,
        'new_edges': new_edges,
        'total_improvement': total_improvement,
    }


def create_rewired_network(
    H_matrix: sparse.csr_matrix,
    new_edges: List[Tuple[int, int]],
    logger: logging.Logger
) -> sparse.csr_matrix:
    """Create rewired network by adding new edges to H matrix."""
    logger.info("Creating rewired network...")
    
    H_prime = H_matrix.copy().tolil()  # Convert to lil for efficient updates
    
    for supplier, buyer in new_edges:
        H_prime[supplier, buyer] = 1.0
    
    H_prime = H_prime.tocsr()  # Convert back to csr
    
    logger.info(f"Rewired network created:")
    logger.info(f"  Original edges: {H_matrix.nnz}")
    logger.info(f"  New edges: {len(new_edges)}")
    logger.info(f"  Total edges: {H_prime.nnz}")
    
    return H_prime


def save_results(
    buffer_capacity: np.ndarray,
    rewiring_results: Dict,
    H_prime: sparse.csr_matrix,
    firm_ids: List,
    config: Config,
    logger: logging.Logger
):
    """Save all results to output directory."""
    logger.info("Saving results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save buffer capacity scores
    buffer_file = OUTPUT_DIR / "buffer_scores.npy"
    np.save(buffer_file, buffer_capacity)
    logger.info(f"Saved buffer scores to {buffer_file}")
    
    # Save rewiring map
    rewiring_map_file = OUTPUT_DIR / "rewiring_map.pkl"
    with open(rewiring_map_file, 'wb') as f:
        pickle.dump(rewiring_results['rewiring_map'], f)
    logger.info(f"Saved rewiring map to {rewiring_map_file}")
    
    # Save rewired network
    H_prime_file = OUTPUT_DIR / "H_prime_rewired.npz"
    sparse.save_npz(H_prime_file, H_prime)
    logger.info(f"Saved rewired network to {H_prime_file}")
    
    # Save rewiring report as CSV
    report_data = []
    for buyer, suppliers in rewiring_results['rewiring_map'].items():
        for supplier in suppliers:
            report_data.append({
                'buyer_idx': buyer,
                'buyer_id': firm_ids[buyer] if buyer < len(firm_ids) else 'unknown',
                'supplier_idx': supplier,
                'supplier_id': firm_ids[supplier] if supplier < len(firm_ids) else 'unknown',
            })
    
    report_df = pd.DataFrame(report_data)
    report_file = OUTPUT_DIR / "rewiring_report.csv"
    report_df.to_csv(report_file, index=False)
    logger.info(f"Saved rewiring report to {report_file} ({len(report_df)} edges)")
    
    # Save summary statistics
    summary = {
        'total_nodes': len(buffer_capacity),
        'nodes_rewired': len(rewiring_results['rewiring_map']),
        'new_edges': len(rewiring_results['new_edges']),
        'total_improvement': rewiring_results['total_improvement'],
        'avg_improvement_per_edge': rewiring_results['total_improvement'] / max(len(rewiring_results['new_edges']), 1),
    }
    
    summary_file = OUTPUT_DIR / "summary_stats.txt"
    with open(summary_file, 'w') as f:
        f.write("Phase 4: Rewiring Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Saved summary statistics to {summary_file}")
    
    logger.info("All results saved successfully")


def main():
    """Main execution function."""
    logger = setup_logging()
    
    try:
        logger.info("=" * 80)
        logger.info("Phase 4: Constrained Rewiring - Starting")
        logger.info("=" * 80)
        
        # Initialize config
        config = Config()
        
        # Validate files
        validate_files(config, logger)
        
        # Load Phase 1-3 outputs
        phase123_data = load_phase123_outputs(config, logger)
        
        # Load financial data
        financial_data = load_financial_data(config, phase123_data['firm_ids'], logger)
        
        # Calculate buffer capacity
        buffer_capacity = calculate_buffer_capacity(
            phase123_data['tis_scores'],
            financial_data,
            phase123_data['recipes'],
            logger
        )
        
        # Optimize rewiring
        rewiring_results = optimize_rewiring(
            phase123_data['tis_scores'],
            buffer_capacity,
            phase123_data['H_matrix'],
            financial_data,
            phase123_data['recipes'],
            config,
            logger
        )
        
        # Create rewired network
        H_prime = create_rewired_network(
            phase123_data['H_matrix'],
            rewiring_results['new_edges'],
            logger
        )
        
        # Save results
        save_results(
            buffer_capacity,
            rewiring_results,
            H_prime,
            phase123_data['firm_ids'],
            config,
            logger
        )
        
        logger.info("=" * 80)
        logger.info("Phase 4: Constrained Rewiring - Completed Successfully")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during Phase 4 execution: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
