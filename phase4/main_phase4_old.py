"""
Phase 4: Constrained Rewiring - Main Execution Script

This script orchestrates the constrained rewiring process:
1. Load temporal graph data and node features
2. Load TIS from Phase 3
3. Configure rewiring parameters
4. Execute rewiring optimization
5. Save rewiring results and analysis

Usage:
    python main_phase4.py --config config/phase4_config.yaml
    python main_phase4.py --year 2024 --method optimization --top_k 100
"""

import os
import sys
import argparse
import logging
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from phase4.src.buffer_calculator import BufferCalculator
from phase4.src.penalty_calculator import PenaltyCalculator
from phase4.src.rewiring_optimizer import RewiringOptimizer
from phase4.src.constraint_checker import ConstraintChecker
from phase4.src.benchmarks import GreedyRewiring, RandomRewiring, TISOptimizedRewiring
from phase4.src.evaluate_rewiring import RewiringEvaluator, batch_evaluate, create_summary_report


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"phase4_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or return defaults."""
    default_config = {
        'data': {
            'year': 2024,
            'tg_file': 'data/processed/tg_{year}_filtered.csv',
            'node_features_file': 'data/processed/posco_network_capital_consumergoods_removed_{year}.csv',
            'tis_file': 'phase3/output/tis_scores_{year}.csv',
        },
        'rewiring': {
            'method': 'optimization',  # 'optimization', 'greedy', 'random', 'tis_optimized'
            'top_k': 100,
            'max_new_edges': 50,
            'shock_threshold': 0.3,
            'alpha': 0.5,
            'beta': 0.3,
            'gamma': 0.2,
        },
        'constraints': {
            'max_supplier_outdegree': 10,
            'max_buyer_indegree': 10,
            'recipe_similarity_threshold': 0.7,
            'capacity_ratio_min': 0.5,
            'capacity_ratio_max': 2.0,
        },
        'output': {
            'output_dir': 'phase4/output',
            'save_intermediate': True,
            'generate_reports': True,
        },
        'logging': {
            'log_dir': 'phase4/logs',
            'log_level': 'INFO',
        },
        'evaluation': {
            'run_baselines': True,
            'pareto_analysis': True,
            'key_metrics': ['weighted_avg_tis', 'high_risk_coverage', 'num_edges', 'density'],
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            # Deep merge
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value
    
    return default_config


def load_data(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Load all required data for Phase 4."""
    year = config['data']['year']
    logger.info(f"Loading data for year {year}...")
    
    data = {}
    
    # Load temporal graph
    tg_file = config['data']['tg_file'].format(year=year)
    if not os.path.exists(tg_file):
        raise FileNotFoundError(f"Temporal graph file not found: {tg_file}")
    
    data['tg_df'] = pd.read_csv(tg_file)
    logger.info(f"Loaded temporal graph: {len(data['tg_df'])} edges")
    
    # Load node features
    node_features_file = config['data']['node_features_file'].format(year=year)
    if not os.path.exists(node_features_file):
        raise FileNotFoundError(f"Node features file not found: {node_features_file}")
    
    data['node_features_df'] = pd.read_csv(node_features_file)
    logger.info(f"Loaded node features: {len(data['node_features_df'])} nodes")
    
    # Load TIS scores
    tis_file = config['data']['tis_file'].format(year=year)
    if not os.path.exists(tis_file):
        raise FileNotFoundError(f"TIS file not found: {tis_file}")
    
    data['tis_df'] = pd.read_csv(tis_file)
    logger.info(f"Loaded TIS scores: {len(data['tis_df'])} nodes")
    
    return data


def prepare_inputs(data: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Prepare inputs for rewiring optimization."""
    logger.info("Preparing inputs for rewiring...")
    
    # Extract TIS scores as numpy array
    tis_col = None
    for col in ['tis', 'TIS', 'tis_score', 'TIS_score']:
        if col in data['tis_df'].columns:
            tis_col = col
            break
    
    if tis_col is None:
        raise ValueError(f"TIS column not found in TIS data. Columns: {data['tis_df'].columns.tolist()}")
    
    tis_scores = data['tis_df'][tis_col].values
    logger.info(f"TIS scores range: [{tis_scores.min():.4f}, {tis_scores.max():.4f}]")
    
    # Extract edge list from temporal graph
    src_col, dst_col = None, None
    for src_candidate in ['src', 'source', 'supplier', 'from']:
        if src_candidate in data['tg_df'].columns:
            src_col = src_candidate
            break
    
    for dst_candidate in ['dst', 'destination', 'buyer', 'to']:
        if dst_candidate in data['tg_df'].columns:
            dst_col = dst_candidate
            break
    
    if src_col is None or dst_col is None:
        raise ValueError(f"Edge columns not found. Columns: {data['tg_df'].columns.tolist()}")
    
    edge_index = np.array([
        data['tg_df'][src_col].values,
        data['tg_df'][dst_col].values
    ])
    logger.info(f"Edge index shape: {edge_index.shape}")
    
    # Extract node features
    feature_cols = [col for col in data['node_features_df'].columns 
                   if col not in ['node_id', 'company_name', 'year', 'sector', 'industry']]
    
    if not feature_cols:
        logger.warning("No feature columns found, using dummy features")
        node_features = np.ones((len(data['node_features_df']), 10))
    else:
        node_features = data['node_features_df'][feature_cols].values
        # Handle NaN values
        node_features = np.nan_to_num(node_features, nan=0.0)
    
    logger.info(f"Node features shape: {node_features.shape}")
    
    # Get top-k vulnerable nodes
    top_k = config['rewiring']['top_k']
    top_k_indices = np.argsort(tis_scores)[-top_k:]
    logger.info(f"Selected top {top_k} vulnerable nodes (TIS range: [{tis_scores[top_k_indices].min():.4f}, {tis_scores[top_k_indices].max():.4f}])")
    
    return {
        'tis_scores': tis_scores,
        'edge_index': edge_index,
        'node_features': node_features,
        'top_k_indices': top_k_indices,
        'node_df': data['node_features_df'],
        'tg_df': data['tg_df'],
        'tis_df': data['tis_df'],
    }


def run_rewiring(inputs: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Execute rewiring optimization."""
    method = config['rewiring']['method']
    logger.info(f"Running rewiring method: {method}")
    
    start_time = time.time()
    
    # Initialize calculators
    buffer_calc = BufferCalculator(
        edge_index=inputs['edge_index'],
        shock_threshold=config['rewiring']['shock_threshold']
    )
    
    penalty_calc = PenaltyCalculator(
        node_features=inputs['node_features'],
        node_df=inputs['node_df']
    )
    
    constraint_checker = ConstraintChecker(
        edge_index=inputs['edge_index'],
        node_features=inputs['node_features'],
        node_df=inputs['node_df'],
        max_supplier_outdegree=config['constraints']['max_supplier_outdegree'],
        max_buyer_indegree=config['constraints']['max_buyer_indegree'],
        recipe_similarity_threshold=config['constraints']['recipe_similarity_threshold'],
        capacity_ratio_min=config['constraints']['capacity_ratio_min'],
        capacity_ratio_max=config['constraints']['capacity_ratio_max']
    )
    
    # Run appropriate method
    if method == 'optimization':
        optimizer = RewiringOptimizer(
            buffer_calculator=buffer_calc,
            penalty_calculator=penalty_calc,
            constraint_checker=constraint_checker,
            alpha=config['rewiring']['alpha'],
            beta=config['rewiring']['beta'],
            gamma=config['rewiring']['gamma']
        )
        
        results = optimizer.optimize_rewiring(
            vulnerable_nodes=inputs['top_k_indices'],
            tis_scores=inputs['tis_scores'],
            max_new_edges=config['rewiring']['max_new_edges']
        )
        
    elif method == 'greedy':
        greedy = GreedyRewiring(
            buffer_calculator=buffer_calc,
            penalty_calculator=penalty_calc,
            constraint_checker=constraint_checker
        )
        
        results = greedy.rewire(
            vulnerable_nodes=inputs['top_k_indices'],
            tis_scores=inputs['tis_scores'],
            max_new_edges=config['rewiring']['max_new_edges']
        )
        
    elif method == 'random':
        random_rewiring = RandomRewiring(
            constraint_checker=constraint_checker,
            num_nodes=inputs['node_features'].shape[0]
        )
        
        results = random_rewiring.rewire(
            vulnerable_nodes=inputs['top_k_indices'],
            max_new_edges=config['rewiring']['max_new_edges']
        )
        
    elif method == 'tis_optimized':
        tis_optimized = TISOptimizedRewiring(
            buffer_calculator=buffer_calc,
            constraint_checker=constraint_checker
        )
        
        results = tis_optimized.rewire(
            vulnerable_nodes=inputs['top_k_indices'],
            tis_scores=inputs['tis_scores'],
            max_new_edges=config['rewiring']['max_new_edges']
        )
        
    else:
        raise ValueError(f"Unknown rewiring method: {method}")
    
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time
    
    logger.info(f"Rewiring completed in {elapsed_time:.2f} seconds")
    logger.info(f"New edges added: {len(results['new_edges'])}")
    logger.info(f"Total improvement: {results.get('total_improvement', 0.0):.4f}")
    
    return results


def save_results(results: Dict[str, Any], inputs: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger):
    """Save rewiring results to files."""
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    year = config['data']['year']
    method = config['rewiring']['method']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save new edges
    if results['new_edges']:
        new_edges_df = pd.DataFrame(results['new_edges'], columns=['supplier', 'buyer'])
        new_edges_file = os.path.join(output_dir, f"new_edges_{method}_{year}_{timestamp}.csv")
        new_edges_df.to_csv(new_edges_file, index=False)
        logger.info(f"Saved new edges to {new_edges_file}")
    
    # Save edge improvements
    if 'edge_improvements' in results and results['edge_improvements']:
        improvements_df = pd.DataFrame(results['edge_improvements'])
        improvements_file = os.path.join(output_dir, f"edge_improvements_{method}_{year}_{timestamp}.csv")
        improvements_df.to_csv(improvements_file, index=False)
        logger.info(f"Saved edge improvements to {improvements_file}")
    
    # Save summary statistics
    summary = {
        'method': method,
        'year': year,
        'timestamp': timestamp,
        'num_new_edges': len(results['new_edges']),
        'total_improvement': results.get('total_improvement', 0.0),
        'avg_improvement': results.get('avg_improvement', 0.0),
        'elapsed_time': results.get('elapsed_time', 0.0),
        'top_k': config['rewiring']['top_k'],
        'max_new_edges': config['rewiring']['max_new_edges'],
    }
    
    if 'objective_history' in results:
        summary['final_objective'] = results['objective_history'][-1] if results['objective_history'] else 0.0
        summary['initial_objective'] = results['objective_history'][0] if results['objective_history'] else 0.0
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_dir, f"summary_{method}_{year}_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to {summary_file}")
    
    # Save full results as pickle (optional)
    if config['output']['save_intermediate']:
        import pickle
        results_file = os.path.join(output_dir, f"full_results_{method}_{year}_{timestamp}.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Saved full results to {results_file}")
    
    logger.info("All results saved successfully")


def evaluate_and_compare(
    results: Dict[str, Any],
    inputs: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Run comprehensive evaluation and comparison with baselines."""
    if not config['output'].get('generate_reports', False):
        logger.info("Report generation disabled, skipping evaluation")
        return None
    
    logger.info("=" * 80)
    logger.info("Running evaluation and comparison")
    logger.info("=" * 80)
    
    # Initialize evaluator
    evaluator = RewiringEvaluator(
        original_edge_index=inputs['edge_index'],
        node_features=inputs['node_features'],
        tis_scores=inputs['tis_scores']
    )
    
    # Collect rewiring results
    rewiring_results = {
        config['rewiring']['method']: results['new_edges']
    }
    
    # Run baselines if enabled
    if config['evaluation'].get('run_baselines', False):
        logger.info("Running baseline methods for comparison...")
        
        # Initialize shared components
        buffer_calc = BufferCalculator(
            edge_index=inputs['edge_index'],
            shock_threshold=config['rewiring']['shock_threshold']
        )
        penalty_calc = PenaltyCalculator(
            node_features=inputs['node_features'],
            node_df=inputs['node_df']
        )
        constraint_checker = ConstraintChecker(
            edge_index=inputs['edge_index'],
            node_features=inputs['node_features'],
            node_df=inputs['node_df'],
            max_supplier_outdegree=config['constraints']['max_supplier_outdegree'],
            max_buyer_indegree=config['constraints']['max_buyer_indegree'],
            recipe_similarity_threshold=config['constraints']['recipe_similarity_threshold'],
            capacity_ratio_min=config['constraints']['capacity_ratio_min'],
            capacity_ratio_max=config['constraints']['capacity_ratio_max']
        )
        
        # Run baselines
        if config['rewiring']['method'] != 'greedy':
            logger.info("  Running Greedy baseline...")
            greedy = GreedyRewiring(buffer_calc, penalty_calc, constraint_checker)
            greedy_results = greedy.rewire(
                inputs['top_k_indices'],
                inputs['tis_scores'],
                config['rewiring']['max_new_edges']
            )
            rewiring_results['greedy'] = greedy_results['new_edges']
        
        if config['rewiring']['method'] != 'random':
            logger.info("  Running Random baseline...")
            random_rewiring = RandomRewiring(constraint_checker, inputs['node_features'].shape[0])
            random_results = random_rewiring.rewire(
                inputs['top_k_indices'],
                config['rewiring']['max_new_edges']
            )
            rewiring_results['random'] = random_results['new_edges']
        
        if config['rewiring']['method'] != 'tis_optimized':
            logger.info("  Running TIS-Optimized baseline...")
            tis_optimized = TISOptimizedRewiring(buffer_calc, constraint_checker)
            tis_results = tis_optimized.rewire(
                inputs['top_k_indices'],
                inputs['tis_scores'],
                config['rewiring']['max_new_edges']
            )
            rewiring_results['tis_optimized'] = tis_results['new_edges']
    
    # Run batch evaluation
    output_dir = os.path.join(config['output']['output_dir'], 'evaluation')
    evaluation_results = batch_evaluate(evaluator, rewiring_results, output_dir)
    
    # Create summary report
    year = config['data']['year']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"summary_report_{year}_{timestamp}.txt")
    create_summary_report(evaluation_results, summary_file)
    
    # Log key findings
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    comparison_df = evaluation_results['comparison']
    logger.info(f"Methods compared: {len(comparison_df)}")
    
    if 'tis_rank' in comparison_df.columns:
        best_tis = comparison_df.loc[comparison_df['tis_rank'] == 1, 'method'].values[0]
        logger.info(f"Best method for TIS improvement: {best_tis}")
    
    if 'coverage_rank' in comparison_df.columns:
        best_coverage = comparison_df.loc[comparison_df['coverage_rank'] == 1, 'method'].values[0]
        logger.info(f"Best method for coverage improvement: {best_coverage}")
    
    pareto_count = len(evaluation_results['pareto_optimal'])
    logger.info(f"Pareto-optimal solutions: {pareto_count}")
    
    logger.info("=" * 80)
    
    return evaluation_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Phase 4: Constrained Rewiring")
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--year', type=int, help='Year to process')
    parser.add_argument('--method', type=str, choices=['optimization', 'greedy', 'random', 'tis_optimized'],
                       help='Rewiring method')
    parser.add_argument('--top_k', type=int, help='Number of top vulnerable nodes')
    parser.add_argument('--max_new_edges', type=int, help='Maximum number of new edges')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.year:
        config['data']['year'] = args.year
    if args.method:
        config['rewiring']['method'] = args.method
    if args.top_k:
        config['rewiring']['top_k'] = args.top_k
    if args.max_new_edges:
        config['rewiring']['max_new_edges'] = args.max_new_edges
    if args.log_level:
        config['logging']['log_level'] = args.log_level
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'], config['logging']['log_level'])
    
    try:
        logger.info("=" * 80)
        logger.info("Phase 4: Constrained Rewiring - Starting")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config}")
        
        # Load data
        data = load_data(config, logger)
        
        # Prepare inputs
        inputs = prepare_inputs(data, config, logger)
        
        # Run rewiring
        results = run_rewiring(inputs, config, logger)
        
        # Save results
        save_results(results, inputs, config, logger)
        
        # Evaluate and compare methods
        evaluation_results = evaluate_and_compare(results, inputs, config, logger)
        
        logger.info("=" * 80)
        logger.info("Phase 4: Constrained Rewiring - Completed Successfully")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during Phase 4 execution: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
