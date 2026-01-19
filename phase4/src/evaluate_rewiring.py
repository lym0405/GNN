"""
Phase 4: Rewiring Evaluation Module

This module provides comprehensive evaluation capabilities for rewiring results:
1. Compute network metrics (resilience, connectivity, efficiency)
2. Compare multiple rewiring methods
3. Generate evaluation reports and visualizations
4. Analyze trade-offs between objectives

Author: Phase 4 Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RewiringEvaluator:
    """
    Evaluator for comparing and analyzing rewiring results.
    
    Metrics computed:
    - Network resilience (TIS-based)
    - Connectivity metrics (degree distribution, path lengths)
    - Efficiency metrics (buffer coverage, constraint satisfaction)
    - Cost metrics (penalties, new edge count)
    """
    
    def __init__(
        self,
        original_edge_index: np.ndarray,
        node_features: np.ndarray,
        tis_scores: np.ndarray
    ):
        """
        Initialize evaluator.
        
        Args:
            original_edge_index: Original network edges [2, num_edges]
            node_features: Node feature matrix [num_nodes, num_features]
            tis_scores: TIS scores for all nodes [num_nodes]
        """
        self.original_edge_index = original_edge_index
        self.node_features = node_features
        self.tis_scores = tis_scores
        self.num_nodes = node_features.shape[0]
        
        # Pre-compute original network metrics
        self.original_metrics = self._compute_network_metrics(original_edge_index)
        logger.info(f"Initialized evaluator with {self.num_nodes} nodes, "
                   f"{original_edge_index.shape[1]} edges")
    
    def _compute_network_metrics(self, edge_index: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive network metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['num_edges'] = edge_index.shape[1]
        metrics['avg_degree'] = edge_index.shape[1] * 2.0 / self.num_nodes
        
        # Degree distribution
        in_degree = np.bincount(edge_index[1], minlength=self.num_nodes)
        out_degree = np.bincount(edge_index[0], minlength=self.num_nodes)
        
        metrics['avg_in_degree'] = in_degree.mean()
        metrics['avg_out_degree'] = out_degree.mean()
        metrics['max_in_degree'] = in_degree.max()
        metrics['max_out_degree'] = out_degree.max()
        metrics['degree_std'] = (in_degree + out_degree).std()
        
        # Resilience: Average TIS weighted by degree
        total_degree = in_degree + out_degree
        if total_degree.sum() > 0:
            weighted_tis = (self.tis_scores * total_degree).sum() / total_degree.sum()
            metrics['weighted_avg_tis'] = weighted_tis
        else:
            metrics['weighted_avg_tis'] = self.tis_scores.mean()
        
        # High-risk node coverage (nodes with TIS > 75th percentile)
        tis_threshold = np.percentile(self.tis_scores, 75)
        high_risk_nodes = np.where(self.tis_scores > tis_threshold)[0]
        high_risk_covered = np.sum(np.isin(high_risk_nodes, edge_index[1]))
        metrics['high_risk_coverage'] = high_risk_covered / len(high_risk_nodes) if len(high_risk_nodes) > 0 else 0.0
        
        # Network density
        max_edges = self.num_nodes * (self.num_nodes - 1)
        metrics['density'] = edge_index.shape[1] / max_edges if max_edges > 0 else 0.0
        
        return metrics
    
    def evaluate_rewiring(
        self,
        new_edges: List[Tuple[int, int]],
        method_name: str = "rewiring"
    ) -> Dict[str, Any]:
        """
        Evaluate a single rewiring result.
        
        Args:
            new_edges: List of new edges [(src, dst), ...]
            method_name: Name of the rewiring method
            
        Returns:
            Dictionary with evaluation metrics and comparisons
        """
        logger.info(f"Evaluating {method_name} with {len(new_edges)} new edges")
        
        # Create new edge index
        if new_edges:
            new_edge_array = np.array(new_edges).T  # [2, num_new_edges]
            rewired_edge_index = np.concatenate([self.original_edge_index, new_edge_array], axis=1)
        else:
            rewired_edge_index = self.original_edge_index
        
        # Compute metrics for rewired network
        rewired_metrics = self._compute_network_metrics(rewired_edge_index)
        
        # Compute improvements
        improvements = {}
        for key in rewired_metrics:
            if key in self.original_metrics:
                original_val = self.original_metrics[key]
                rewired_val = rewired_metrics[key]
                
                # For most metrics, higher is better; for TIS, lower is better
                if 'tis' in key.lower():
                    improvement = original_val - rewired_val  # Reduction in TIS is good
                else:
                    improvement = rewired_val - original_val
                
                improvements[f'{key}_improvement'] = improvement
                improvements[f'{key}_relative_improvement'] = improvement / abs(original_val) if original_val != 0 else 0.0
        
        # Analyze new edges
        edge_analysis = self._analyze_new_edges(new_edges)
        
        # Compile results
        results = {
            'method': method_name,
            'num_new_edges': len(new_edges),
            'original_metrics': self.original_metrics,
            'rewired_metrics': rewired_metrics,
            'improvements': improvements,
            'edge_analysis': edge_analysis,
        }
        
        return results
    
    def _analyze_new_edges(self, new_edges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze properties of new edges."""
        if not new_edges:
            return {
                'avg_supplier_tis': 0.0,
                'avg_buyer_tis': 0.0,
                'high_tis_buyer_count': 0,
                'avg_tis_reduction_potential': 0.0,
            }
        
        suppliers = [src for src, _ in new_edges]
        buyers = [dst for _, dst in new_edges]
        
        supplier_tis = self.tis_scores[suppliers]
        buyer_tis = self.tis_scores[buyers]
        
        # TIS reduction potential: how much buyer TIS could be reduced
        tis_threshold = np.percentile(self.tis_scores, 75)
        high_tis_buyers = np.sum(buyer_tis > tis_threshold)
        
        return {
            'avg_supplier_tis': supplier_tis.mean(),
            'avg_buyer_tis': buyer_tis.mean(),
            'high_tis_buyer_count': int(high_tis_buyers),
            'high_tis_buyer_ratio': high_tis_buyers / len(buyers),
            'avg_tis_reduction_potential': np.maximum(0, buyer_tis - supplier_tis).mean(),
        }
    
    def compare_methods(
        self,
        results_dict: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple rewiring methods.
        
        Args:
            results_dict: Dictionary mapping method names to evaluation results
            
        Returns:
            DataFrame with comparison metrics
        """
        logger.info(f"Comparing {len(results_dict)} methods")
        
        comparison_data = []
        
        for method_name, results in results_dict.items():
            row = {
                'method': method_name,
                'num_new_edges': results['num_new_edges'],
            }
            
            # Add key metrics
            for metric_name, value in results['rewired_metrics'].items():
                row[f'rewired_{metric_name}'] = value
            
            # Add key improvements
            for metric_name, value in results['improvements'].items():
                if 'relative_improvement' in metric_name:
                    row[metric_name] = value
            
            # Add edge analysis
            for metric_name, value in results['edge_analysis'].items():
                row[f'edge_{metric_name}'] = value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank methods by key metrics
        if 'weighted_avg_tis_relative_improvement' in comparison_df.columns:
            comparison_df['tis_rank'] = comparison_df['weighted_avg_tis_relative_improvement'].rank(ascending=False)
        
        if 'high_risk_coverage_relative_improvement' in comparison_df.columns:
            comparison_df['coverage_rank'] = comparison_df['high_risk_coverage_relative_improvement'].rank(ascending=False)
        
        return comparison_df
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a detailed evaluation report.
        
        Args:
            results: Evaluation results from evaluate_rewiring()
            output_file: Optional path to save report
            
        Returns:
            Report as a string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"REWIRING EVALUATION REPORT: {results['method']}")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Number of new edges added: {results['num_new_edges']}")
        lines.append("")
        
        # Original network metrics
        lines.append("ORIGINAL NETWORK METRICS")
        lines.append("-" * 80)
        for key, value in results['original_metrics'].items():
            lines.append(f"  {key:30s}: {value:12.6f}")
        lines.append("")
        
        # Rewired network metrics
        lines.append("REWIRED NETWORK METRICS")
        lines.append("-" * 80)
        for key, value in results['rewired_metrics'].items():
            lines.append(f"  {key:30s}: {value:12.6f}")
        lines.append("")
        
        # Improvements
        lines.append("IMPROVEMENTS")
        lines.append("-" * 80)
        for key, value in results['improvements'].items():
            if 'relative' in key:
                lines.append(f"  {key:40s}: {value:12.2%}")
            else:
                lines.append(f"  {key:40s}: {value:12.6f}")
        lines.append("")
        
        # Edge analysis
        lines.append("NEW EDGE ANALYSIS")
        lines.append("-" * 80)
        for key, value in results['edge_analysis'].items():
            if isinstance(value, float):
                lines.append(f"  {key:40s}: {value:12.6f}")
            else:
                lines.append(f"  {key:40s}: {value}")
        lines.append("")
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    def compute_pareto_frontier(
        self,
        results_list: List[Dict[str, Any]],
        objective1_key: str = 'weighted_avg_tis',
        objective2_key: str = 'num_new_edges'
    ) -> List[Dict[str, Any]]:
        """
        Compute Pareto frontier for multi-objective comparison.
        
        Args:
            results_list: List of evaluation results
            objective1_key: Key for first objective (to minimize)
            objective2_key: Key for second objective (to minimize)
            
        Returns:
            List of Pareto-optimal solutions
        """
        pareto_optimal = []
        
        for i, result_i in enumerate(results_list):
            is_dominated = False
            
            # Extract objectives
            if objective1_key in result_i['rewired_metrics']:
                obj1_i = result_i['rewired_metrics'][objective1_key]
            elif objective1_key in result_i:
                obj1_i = result_i[objective1_key]
            else:
                continue
            
            if objective2_key in result_i['rewired_metrics']:
                obj2_i = result_i['rewired_metrics'][objective2_key]
            elif objective2_key in result_i:
                obj2_i = result_i[objective2_key]
            else:
                continue
            
            # Check if dominated by any other solution
            for j, result_j in enumerate(results_list):
                if i == j:
                    continue
                
                if objective1_key in result_j['rewired_metrics']:
                    obj1_j = result_j['rewired_metrics'][objective1_key]
                elif objective1_key in result_j:
                    obj1_j = result_j[objective1_key]
                else:
                    continue
                
                if objective2_key in result_j['rewired_metrics']:
                    obj2_j = result_j['rewired_metrics'][objective2_key]
                elif objective2_key in result_j:
                    obj2_j = result_j[objective2_key]
                else:
                    continue
                
                # Check dominance (both objectives should be minimized)
                if obj1_j <= obj1_i and obj2_j <= obj2_i and (obj1_j < obj1_i or obj2_j < obj2_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(result_i)
        
        logger.info(f"Found {len(pareto_optimal)} Pareto-optimal solutions out of {len(results_list)}")
        return pareto_optimal


def batch_evaluate(
    evaluator: RewiringEvaluator,
    rewiring_results: Dict[str, List[Tuple[int, int]]],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Batch evaluate multiple rewiring methods.
    
    Args:
        evaluator: RewiringEvaluator instance
        rewiring_results: Dict mapping method names to new edge lists
        output_dir: Optional directory to save reports
        
    Returns:
        Dictionary with all evaluation results and comparisons
    """
    import os
    
    all_results = {}
    
    # Evaluate each method
    for method_name, new_edges in rewiring_results.items():
        logger.info(f"Evaluating method: {method_name}")
        results = evaluator.evaluate_rewiring(new_edges, method_name)
        all_results[method_name] = results
        
        # Generate individual report
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_file = os.path.join(output_dir, f"report_{method_name}.txt")
            evaluator.generate_report(results, report_file)
    
    # Compare methods
    comparison_df = evaluator.compare_methods(all_results)
    
    if output_dir:
        comparison_file = os.path.join(output_dir, "method_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Comparison saved to {comparison_file}")
    
    # Compute Pareto frontier
    results_list = list(all_results.values())
    pareto_optimal = evaluator.compute_pareto_frontier(results_list)
    
    return {
        'individual_results': all_results,
        'comparison': comparison_df,
        'pareto_optimal': pareto_optimal,
    }


def create_summary_report(
    batch_results: Dict[str, Any],
    output_file: str
):
    """
    Create a comprehensive summary report for all methods.
    
    Args:
        batch_results: Results from batch_evaluate()
        output_file: Path to save the summary report
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REWIRING METHODS COMPARISON - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Method rankings
        f.write("METHOD RANKINGS\n")
        f.write("-" * 80 + "\n")
        comparison_df = batch_results['comparison']
        
        if 'tis_rank' in comparison_df.columns:
            f.write("\nBy TIS Improvement:\n")
            ranked = comparison_df.sort_values('tis_rank')
            for idx, row in ranked.iterrows():
                f.write(f"  {int(row['tis_rank'])}. {row['method']:20s} "
                       f"(Improvement: {row.get('weighted_avg_tis_relative_improvement', 0):.2%})\n")
        
        if 'coverage_rank' in comparison_df.columns:
            f.write("\nBy Coverage Improvement:\n")
            ranked = comparison_df.sort_values('coverage_rank')
            for idx, row in ranked.iterrows():
                f.write(f"  {int(row['coverage_rank'])}. {row['method']:20s} "
                       f"(Improvement: {row.get('high_risk_coverage_relative_improvement', 0):.2%})\n")
        
        # Pareto optimal solutions
        f.write("\n" + "=" * 80 + "\n")
        f.write("PARETO OPTIMAL SOLUTIONS\n")
        f.write("-" * 80 + "\n")
        for solution in batch_results['pareto_optimal']:
            f.write(f"  Method: {solution['method']}\n")
            f.write(f"    New edges: {solution['num_new_edges']}\n")
            f.write(f"    Weighted TIS: {solution['rewired_metrics'].get('weighted_avg_tis', 0):.6f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    logger.info(f"Summary report saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    print("RewiringEvaluator module loaded successfully")
    print("Use this module to evaluate and compare rewiring results")
