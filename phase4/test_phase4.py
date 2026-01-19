"""
Phase 4: Testing Script

This script validates all Phase 4 modules with synthetic data.
Run this before executing on real data to ensure all components work correctly.

Usage:
    python test_phase4.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from phase4.src.buffer_calculator import BufferCalculator
from phase4.src.penalty_calculator import PenaltyCalculator
from phase4.src.rewiring_optimizer import RewiringOptimizer
from phase4.src.constraint_checker import ConstraintChecker
from phase4.src.benchmarks import GreedyRewiring, RandomRewiring, TISOptimizedRewiring
from phase4.src.evaluate_rewiring import RewiringEvaluator, batch_evaluate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_data(num_nodes=100, num_edges=200, seed=42):
    """Create synthetic data for testing."""
    np.random.seed(seed)
    
    # Create random edge index
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    for i in range(num_edges):
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        while src == dst:  # Avoid self-loops
            dst = np.random.randint(0, num_nodes)
        edge_index[0, i] = src
        edge_index[1, i] = dst
    
    # Create node features
    node_features = np.random.randn(num_nodes, 20)
    
    # Create node DataFrame with financial and recipe data
    node_df = pd.DataFrame({
        'node_id': range(num_nodes),
        'company_name': [f'Company_{i}' for i in range(num_nodes)],
        'total_assets': np.random.uniform(1e6, 1e9, num_nodes),
        'revenue': np.random.uniform(1e5, 1e8, num_nodes),
        'equity': np.random.uniform(5e5, 5e8, num_nodes),
        'debt': np.random.uniform(1e5, 3e8, num_nodes),
        'sector': np.random.choice(['Manufacturing', 'Services', 'Technology'], num_nodes),
    })
    
    # Add recipe columns (one-hot encoded industry categories)
    for i in range(5):
        node_df[f'recipe_{i}'] = np.random.randint(0, 2, num_nodes)
    
    # Create TIS scores (higher = more vulnerable)
    tis_scores = np.random.uniform(0.0, 1.0, num_nodes)
    
    return edge_index, node_features, node_df, tis_scores


def test_buffer_calculator():
    """Test BufferCalculator module."""
    logger.info("Testing BufferCalculator...")
    
    edge_index, node_features, node_df, tis_scores = create_synthetic_data()
    
    calc = BufferCalculator(edge_index=edge_index, shock_threshold=0.3)
    
    # Test buffer calculation for a few nodes
    test_nodes = [0, 10, 20]
    test_suppliers = [5, 15, 25]
    
    for node, supplier in zip(test_nodes, test_suppliers):
        buffer = calc.compute_buffer(node, supplier, tis_scores, node_df)
        assert buffer >= 0.0, f"Buffer should be non-negative, got {buffer}"
        logger.info(f"  Node {node} + Supplier {supplier}: Buffer = {buffer:.4f}")
    
    logger.info("✓ BufferCalculator test passed\n")


def test_penalty_calculator():
    """Test PenaltyCalculator module."""
    logger.info("Testing PenaltyCalculator...")
    
    edge_index, node_features, node_df, tis_scores = create_synthetic_data()
    
    calc = PenaltyCalculator(node_features=node_features, node_df=node_df)
    
    # Test penalty calculation
    test_pairs = [(0, 5), (10, 15), (20, 25)]
    
    for supplier, buyer in test_pairs:
        penalty = calc.compute_penalty(supplier, buyer)
        assert penalty >= 0.0, f"Penalty should be non-negative, got {penalty}"
        logger.info(f"  Edge ({supplier}, {buyer}): Penalty = {penalty:.4f}")
    
    logger.info("✓ PenaltyCalculator test passed\n")


def test_constraint_checker():
    """Test ConstraintChecker module."""
    logger.info("Testing ConstraintChecker...")
    
    edge_index, node_features, node_df, tis_scores = create_synthetic_data()
    
    checker = ConstraintChecker(
        edge_index=edge_index,
        node_features=node_features,
        node_df=node_df,
        max_supplier_outdegree=10,
        max_buyer_indegree=10,
        recipe_similarity_threshold=0.7,
        capacity_ratio_min=0.5,
        capacity_ratio_max=2.0
    )
    
    # Test individual constraints
    test_edges = [(0, 5), (10, 15), (20, 25), (0, 0)]  # Last one is self-loop
    
    for supplier, buyer in test_edges:
        is_valid = checker.check_constraints(supplier, buyer)
        logger.info(f"  Edge ({supplier}, {buyer}): Valid = {is_valid}")
        
        if supplier == buyer:
            assert not is_valid, "Self-loop should be invalid"
    
    logger.info("✓ ConstraintChecker test passed\n")


def test_rewiring_optimizer():
    """Test RewiringOptimizer module."""
    logger.info("Testing RewiringOptimizer...")
    
    edge_index, node_features, node_df, tis_scores = create_synthetic_data()
    
    # Initialize components
    buffer_calc = BufferCalculator(edge_index=edge_index, shock_threshold=0.3)
    penalty_calc = PenaltyCalculator(node_features=node_features, node_df=node_df)
    constraint_checker = ConstraintChecker(
        edge_index=edge_index,
        node_features=node_features,
        node_df=node_df,
        max_supplier_outdegree=10,
        max_buyer_indegree=10,
        recipe_similarity_threshold=0.7,
        capacity_ratio_min=0.5,
        capacity_ratio_max=2.0
    )
    
    optimizer = RewiringOptimizer(
        buffer_calculator=buffer_calc,
        penalty_calculator=penalty_calc,
        constraint_checker=constraint_checker,
        alpha=0.5,
        beta=0.3,
        gamma=0.2
    )
    
    # Run optimization
    vulnerable_nodes = np.argsort(tis_scores)[-10:]  # Top 10 vulnerable
    results = optimizer.optimize_rewiring(
        vulnerable_nodes=vulnerable_nodes,
        tis_scores=tis_scores,
        max_new_edges=5
    )
    
    assert 'new_edges' in results, "Results should contain new_edges"
    assert 'total_improvement' in results, "Results should contain total_improvement"
    assert len(results['new_edges']) <= 5, "Should not exceed max_new_edges"
    
    logger.info(f"  Added {len(results['new_edges'])} new edges")
    logger.info(f"  Total improvement: {results['total_improvement']:.4f}")
    logger.info("✓ RewiringOptimizer test passed\n")


def test_benchmarks():
    """Test benchmark methods."""
    logger.info("Testing Benchmark Methods...")
    
    edge_index, node_features, node_df, tis_scores = create_synthetic_data()
    
    # Initialize components
    buffer_calc = BufferCalculator(edge_index=edge_index, shock_threshold=0.3)
    penalty_calc = PenaltyCalculator(node_features=node_features, node_df=node_df)
    constraint_checker = ConstraintChecker(
        edge_index=edge_index,
        node_features=node_features,
        node_df=node_df,
        max_supplier_outdegree=10,
        max_buyer_indegree=10,
        recipe_similarity_threshold=0.7,
        capacity_ratio_min=0.5,
        capacity_ratio_max=2.0
    )
    
    vulnerable_nodes = np.argsort(tis_scores)[-10:]
    max_new_edges = 5
    
    # Test Greedy
    logger.info("  Testing Greedy...")
    greedy = GreedyRewiring(buffer_calc, penalty_calc, constraint_checker)
    greedy_results = greedy.rewire(vulnerable_nodes, tis_scores, max_new_edges)
    assert len(greedy_results['new_edges']) <= max_new_edges
    logger.info(f"    Added {len(greedy_results['new_edges'])} edges")
    
    # Test Random
    logger.info("  Testing Random...")
    random_rewiring = RandomRewiring(constraint_checker, node_features.shape[0])
    random_results = random_rewiring.rewire(vulnerable_nodes, max_new_edges)
    assert len(random_results['new_edges']) <= max_new_edges
    logger.info(f"    Added {len(random_results['new_edges'])} edges")
    
    # Test TIS-Optimized
    logger.info("  Testing TIS-Optimized...")
    tis_optimized = TISOptimizedRewiring(buffer_calc, constraint_checker)
    tis_results = tis_optimized.rewire(vulnerable_nodes, tis_scores, max_new_edges)
    assert len(tis_results['new_edges']) <= max_new_edges
    logger.info(f"    Added {len(tis_results['new_edges'])} edges")
    
    logger.info("✓ Benchmark methods test passed\n")


def test_evaluator():
    """Test RewiringEvaluator module."""
    logger.info("Testing RewiringEvaluator...")
    
    edge_index, node_features, node_df, tis_scores = create_synthetic_data()
    
    evaluator = RewiringEvaluator(
        original_edge_index=edge_index,
        node_features=node_features,
        tis_scores=tis_scores
    )
    
    # Create some test rewiring results
    new_edges_1 = [(0, 5), (10, 15), (20, 25)]
    new_edges_2 = [(1, 6), (11, 16)]
    
    results_1 = evaluator.evaluate_rewiring(new_edges_1, "test_method_1")
    results_2 = evaluator.evaluate_rewiring(new_edges_2, "test_method_2")
    
    assert 'original_metrics' in results_1
    assert 'rewired_metrics' in results_1
    assert 'improvements' in results_1
    
    logger.info(f"  Method 1: {len(new_edges_1)} edges, "
               f"TIS: {results_1['rewired_metrics']['weighted_avg_tis']:.4f}")
    logger.info(f"  Method 2: {len(new_edges_2)} edges, "
               f"TIS: {results_2['rewired_metrics']['weighted_avg_tis']:.4f}")
    
    # Test comparison
    comparison_df = evaluator.compare_methods({
        'method_1': results_1,
        'method_2': results_2
    })
    
    assert len(comparison_df) == 2
    logger.info(f"  Comparison DataFrame shape: {comparison_df.shape}")
    
    logger.info("✓ RewiringEvaluator test passed\n")


def run_all_tests():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("PHASE 4 MODULE TESTING")
    logger.info("=" * 80)
    logger.info("")
    
    tests = [
        test_buffer_calculator,
        test_penalty_calculator,
        test_constraint_checker,
        test_rewiring_optimizer,
        test_benchmarks,
        test_evaluator,
    ]
    
    failed_tests = []
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            logger.error(f"✗ {test_func.__name__} FAILED: {str(e)}")
            failed_tests.append(test_func.__name__)
            import traceback
            traceback.print_exc()
    
    logger.info("=" * 80)
    if failed_tests:
        logger.error(f"TESTING COMPLETED WITH {len(failed_tests)} FAILURES")
        for test_name in failed_tests:
            logger.error(f"  - {test_name}")
        return 1
    else:
        logger.info("ALL TESTS PASSED ✓")
        logger.info("Phase 4 modules are ready for use!")
        return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
