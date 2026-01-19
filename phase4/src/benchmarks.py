"""
Phase 4: Benchmark Rewiring Methods

This module implements baseline rewiring methods for comparison:
1. Greedy: Select edges greedily by local benefit
2. Random: Random edge selection with constraints
3. TIS-Optimized: Focused on TIS reduction

Author: Phase 4 Team
Date: 2024
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class GreedyRewiring:
    """
    Greedy rewiring: Select edges by local greedy benefit.
    
    At each step, selects the edge with highest immediate benefit
    (buffer - penalty) that satisfies constraints.
    """
    
    def __init__(
        self,
        buffer_calculator,
        penalty_calculator,
        constraint_checker
    ):
        """
        Initialize Greedy rewiring.
        
        Args:
            buffer_calculator: BufferCalculator instance
            penalty_calculator: PenaltyCalculator instance
            constraint_checker: ConstraintChecker instance
        """
        self.buffer_calc = buffer_calculator
        self.penalty_calc = penalty_calculator
        self.constraint_checker = constraint_checker
        
        logger.info("Initialized GreedyRewiring")
    
    def rewire(
        self,
        vulnerable_nodes: np.ndarray,
        tis_scores: np.ndarray,
        max_new_edges: int
    ) -> Dict[str, Any]:
        """
        Perform greedy rewiring.
        
        Args:
            vulnerable_nodes: Indices of vulnerable nodes [K]
            tis_scores: TIS scores for all nodes [N]
            max_new_edges: Maximum number of new edges
            
        Returns:
            Dictionary with new edges and statistics
        """
        logger.info(f"Starting Greedy rewiring for {len(vulnerable_nodes)} nodes")
        
        new_edges = []
        edge_scores = []
        num_nodes = tis_scores.shape[0]
        
        # For each vulnerable node, find best supplier
        for buyer in vulnerable_nodes:
            if len(new_edges) >= max_new_edges:
                break
            
            best_supplier = None
            best_score = -np.inf
            
            # Try all potential suppliers
            for supplier in range(num_nodes):
                if supplier == buyer:
                    continue
                
                # Check constraints
                if not self.constraint_checker.check_constraints(supplier, buyer):
                    continue
                
                # Compute score
                buffer = self.buffer_calc.compute_buffer(
                    buyer, supplier, tis_scores,
                    self.constraint_checker.node_df
                )
                penalty = self.penalty_calc.compute_penalty(supplier, buyer)
                
                score = buffer - penalty
                
                if score > best_score:
                    best_score = score
                    best_supplier = supplier
            
            if best_supplier is not None:
                new_edges.append((best_supplier, buyer))
                edge_scores.append(best_score)
        
        results = {
            'new_edges': new_edges,
            'edge_scores': edge_scores,
            'total_improvement': sum(edge_scores) if edge_scores else 0.0,
            'avg_improvement': np.mean(edge_scores) if edge_scores else 0.0,
        }
        
        logger.info(f"Greedy rewiring complete: {len(new_edges)} edges added")
        
        return results


class RandomRewiring:
    """
    Random rewiring: Select edges randomly while satisfying constraints.
    
    Baseline method for comparison.
    """
    
    def __init__(
        self,
        constraint_checker,
        num_nodes: int,
        seed: int = 42
    ):
        """
        Initialize Random rewiring.
        
        Args:
            constraint_checker: ConstraintChecker instance
            num_nodes: Total number of nodes
            seed: Random seed
        """
        self.constraint_checker = constraint_checker
        self.num_nodes = num_nodes
        self.rng = np.random.RandomState(seed)
        
        logger.info(f"Initialized RandomRewiring with {num_nodes} nodes")
    
    def rewire(
        self,
        vulnerable_nodes: np.ndarray,
        max_new_edges: int,
        max_attempts_per_node: int = 100
    ) -> Dict[str, Any]:
        """
        Perform random rewiring.
        
        Args:
            vulnerable_nodes: Indices of vulnerable nodes [K]
            max_new_edges: Maximum number of new edges
            max_attempts_per_node: Max attempts to find valid edge per node
            
        Returns:
            Dictionary with new edges and statistics
        """
        logger.info(f"Starting Random rewiring for {len(vulnerable_nodes)} nodes")
        
        new_edges = []
        
        for buyer in vulnerable_nodes:
            if len(new_edges) >= max_new_edges:
                break
            
            # Try random suppliers
            for _ in range(max_attempts_per_node):
                supplier = self.rng.randint(0, self.num_nodes)
                
                if supplier == buyer:
                    continue
                
                # Check constraints
                if self.constraint_checker.check_constraints(supplier, buyer):
                    new_edges.append((supplier, buyer))
                    break
        
        results = {
            'new_edges': new_edges,
            'total_improvement': 0.0,  # Random doesn't optimize
            'avg_improvement': 0.0,
        }
        
        logger.info(f"Random rewiring complete: {len(new_edges)} edges added")
        
        return results


class TISOptimizedRewiring:
    """
    TIS-Optimized rewiring: Focus on connecting vulnerable buyers to low-TIS suppliers.
    
    Prioritizes TIS reduction over other objectives.
    """
    
    def __init__(
        self,
        buffer_calculator,
        constraint_checker
    ):
        """
        Initialize TIS-Optimized rewiring.
        
        Args:
            buffer_calculator: BufferCalculator instance
            constraint_checker: ConstraintChecker instance
        """
        self.buffer_calc = buffer_calculator
        self.constraint_checker = constraint_checker
        
        logger.info("Initialized TISOptimizedRewiring")
    
    def rewire(
        self,
        vulnerable_nodes: np.ndarray,
        tis_scores: np.ndarray,
        max_new_edges: int
    ) -> Dict[str, Any]:
        """
        Perform TIS-optimized rewiring.
        
        Args:
            vulnerable_nodes: Indices of vulnerable nodes [K]
            tis_scores: TIS scores for all nodes [N]
            max_new_edges: Maximum number of new edges
            
        Returns:
            Dictionary with new edges and statistics
        """
        logger.info(f"Starting TIS-Optimized rewiring for {len(vulnerable_nodes)} nodes")
        
        new_edges = []
        tis_reductions = []
        num_nodes = tis_scores.shape[0]
        
        # Sort suppliers by TIS (lowest first)
        supplier_candidates = np.argsort(tis_scores)
        
        # For each vulnerable node, find low-TIS supplier
        for buyer in vulnerable_nodes:
            if len(new_edges) >= max_new_edges:
                break
            
            buyer_tis = tis_scores[buyer]
            
            # Try suppliers in order of TIS
            for supplier in supplier_candidates:
                if supplier == buyer:
                    continue
                
                # Skip if supplier has higher TIS
                if tis_scores[supplier] >= buyer_tis:
                    continue
                
                # Check constraints
                if not self.constraint_checker.check_constraints(supplier, buyer):
                    continue
                
                # Accept this edge
                new_edges.append((supplier, buyer))
                tis_reduction = buyer_tis - tis_scores[supplier]
                tis_reductions.append(tis_reduction)
                break
        
        results = {
            'new_edges': new_edges,
            'tis_reductions': tis_reductions,
            'total_improvement': sum(tis_reductions) if tis_reductions else 0.0,
            'avg_improvement': np.mean(tis_reductions) if tis_reductions else 0.0,
        }
        
        logger.info(f"TIS-Optimized rewiring complete: {len(new_edges)} edges added")
        logger.info(f"  Average TIS reduction: {results['avg_improvement']:.4f}")
        
        return results


def compare_methods(
    buffer_calculator,
    penalty_calculator,
    constraint_checker,
    vulnerable_nodes: np.ndarray,
    tis_scores: np.ndarray,
    max_new_edges: int
) -> Dict[str, Dict[str, Any]]:
    """
    Compare all baseline methods.
    
    Args:
        buffer_calculator: BufferCalculator instance
        penalty_calculator: PenaltyCalculator instance
        constraint_checker: ConstraintChecker instance
        vulnerable_nodes: Indices of vulnerable nodes
        tis_scores: TIS scores for all nodes
        max_new_edges: Maximum number of new edges
        
    Returns:
        Dictionary mapping method names to results
    """
    logger.info("=" * 80)
    logger.info("Comparing all baseline methods")
    logger.info("=" * 80)
    
    num_nodes = tis_scores.shape[0]
    
    methods = {
        'Greedy': GreedyRewiring(buffer_calculator, penalty_calculator, constraint_checker),
        'Random': RandomRewiring(constraint_checker, num_nodes),
        'TIS-Optimized': TISOptimizedRewiring(buffer_calculator, constraint_checker),
    }
    
    results = {}
    
    for method_name, method in methods.items():
        logger.info(f"\nRunning {method_name}...")
        
        if method_name == 'Random':
            result = method.rewire(vulnerable_nodes, max_new_edges)
        else:
            result = method.rewire(vulnerable_nodes, tis_scores, max_new_edges)
        
        results[method_name] = result
        
        logger.info(f"  {method_name}: {len(result['new_edges'])} edges, "
                   f"improvement: {result.get('total_improvement', 0.0):.4f}")
    
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    print("Benchmark methods module loaded successfully")



class GreedyRewiring:
    """
    Greedy 재배선: 링크 확률만 고려
    
    Parameters
    ----------
    link_probs : np.ndarray [N, N]
        링크 예측 확률
    """
    
    def __init__(self, link_probs: np.ndarray):
        self.link_probs = link_probs
        self.num_nodes = link_probs.shape[0]
        
        logger.info("GreedyRewiring 초기화")
        logger.info(f"  - 노드 수: {self.num_nodes:,}")
    
    def rewire(
        self,
        disrupted_nodes: List[int],
        top_k: int = 100,
        min_prob: float = 0.1
    ) -> Dict[int, int]:
        """
        Greedy 재배선 수행
        
        Parameters
        ----------
        disrupted_nodes : list
            단절 노드 리스트
        top_k : int
            후보군 크기
        min_prob : float
            최소 확률
        
        Returns
        -------
        rewiring_map : dict
            재배선 맵
        """
        logger.info("=" * 70)
        logger.info("Greedy 재배선 시작")
        logger.info("=" * 70)
        
        rewiring_map = {}
        
        for i, src in enumerate(disrupted_nodes):
            if (i + 1) % 100 == 0:
                logger.info(f"  진행: {i+1}/{len(disrupted_nodes)}")
            
            # 링크 확률
            probs = self.link_probs[src]
            
            # 최소 임계값 이상
            valid_indices = np.where(probs >= min_prob)[0]
            
            if len(valid_indices) == 0:
                continue
            
            # 최대 확률 선택
            best_target = valid_indices[np.argmax(probs[valid_indices])]
            
            # 자기 루프 방지
            if best_target != src:
                rewiring_map[src] = best_target
        
        logger.info(f"✅ Greedy 재배선 완료: {len(rewiring_map)}/{len(disrupted_nodes)}")
        
        return rewiring_map


class RandomRewiring:
    """
    Random 재배선: 무작위 선택
    
    Parameters
    ----------
    num_nodes : int
        전체 노드 수
    seed : int
        랜덤 시드
    """
    
    def __init__(self, num_nodes: int, seed: int = 42):
        self.num_nodes = num_nodes
        self.rng = np.random.RandomState(seed)
        
        logger.info("RandomRewiring 초기화")
        logger.info(f"  - 노드 수: {num_nodes:,}")
        logger.info(f"  - 시드: {seed}")
    
    def rewire(
        self,
        disrupted_nodes: List[int],
        exclude_self: bool = True
    ) -> Dict[int, int]:
        """
        Random 재배선 수행
        
        Parameters
        ----------
        disrupted_nodes : list
            단절 노드 리스트
        exclude_self : bool
            자기 루프 제외 여부
        
        Returns
        -------
        rewiring_map : dict
            재배선 맵
        """
        logger.info("=" * 70)
        logger.info("Random 재배선 시작")
        logger.info("=" * 70)
        
        rewiring_map = {}
        
        for src in disrupted_nodes:
            # 후보 (자신 제외)
            if exclude_self:
                candidates = [i for i in range(self.num_nodes) if i != src]
            else:
                candidates = list(range(self.num_nodes))
            
            # 무작위 선택
            target = self.rng.choice(candidates)
            rewiring_map[src] = target
        
        logger.info(f"✅ Random 재배선 완료: {len(rewiring_map)}/{len(disrupted_nodes)}")
        
        return rewiring_map


class BenchmarkComparator:
    """
    벤치마크 비교기
    
    Parameters
    ----------
    link_probs : np.ndarray
    buffer_scores : np.ndarray
    """
    
    def __init__(
        self,
        link_probs: np.ndarray,
        buffer_scores: np.ndarray
    ):
        self.link_probs = link_probs
        self.buffer_scores = buffer_scores
        self.num_nodes = link_probs.shape[0]
        
        logger.info("BenchmarkComparator 초기화")
    
    def compare_all(
        self,
        disrupted_nodes: List[int],
        tis_optimized_map: Dict[int, int],
        top_k: int = 100,
        min_prob: float = 0.1
    ) -> Dict[str, Dict]:
        """
        모든 방법 비교
        
        Parameters
        ----------
        disrupted_nodes : list
        tis_optimized_map : dict
            TIS-Optimized 결과
        top_k : int
        min_prob : float
        
        Returns
        -------
        comparison : dict
            비교 결과
        """
        logger.info("=" * 70)
        logger.info("벤치마크 비교 시작")
        logger.info("=" * 70)
        
        # 1. Greedy
        greedy = GreedyRewiring(self.link_probs)
        greedy_map = greedy.rewire(disrupted_nodes, top_k, min_prob)
        
        # 2. Random
        random = RandomRewiring(self.num_nodes)
        random_map = random.rewire(disrupted_nodes)
        
        # 3. 평가
        comparison = {
            'greedy': self._evaluate_rewiring(greedy_map),
            'random': self._evaluate_rewiring(random_map),
            'tis_optimized': self._evaluate_rewiring(tis_optimized_map)
        }
        
        # 결과 출력
        self._print_comparison(comparison)
        
        return comparison
    
    def _evaluate_rewiring(self, rewiring_map: Dict[int, int]) -> Dict:
        """
        재배선 평가
        """
        if len(rewiring_map) == 0:
            return {
                'num_rewired': 0,
                'avg_buffer': 0.0,
                'avg_link_prob': 0.0
            }
        
        targets = list(rewiring_map.values())
        sources = list(rewiring_map.keys())
        
        # Buffer
        avg_buffer = np.mean(self.buffer_scores[targets])
        
        # 링크 확률
        link_probs = [
            self.link_probs[src, tgt]
            for src, tgt in rewiring_map.items()
        ]
        avg_link_prob = np.mean(link_probs)
        
        return {
            'num_rewired': len(rewiring_map),
            'avg_buffer': avg_buffer,
            'avg_link_prob': avg_link_prob
        }
    
    def _print_comparison(self, comparison: Dict):
        """
        비교 결과 출력
        """
        logger.info("=" * 70)
        logger.info("벤치마크 비교 결과")
        logger.info("=" * 70)
        
        methods = ['greedy', 'random', 'tis_optimized']
        method_names = ['Greedy', 'Random', 'TIS-Optimized']
        
        for method, name in zip(methods, method_names):
            result = comparison[method]
            logger.info(f"\n[{name}]")
            logger.info(f"  - 재배선 수: {result['num_rewired']:,}")
            logger.info(f"  - 평균 Buffer: {result['avg_buffer']:.4f}")
            logger.info(f"  - 평균 링크 확률: {result['avg_link_prob']:.4f}")
        
        # 상대 성능
        logger.info("\n[상대 성능 (vs Greedy)]")
        greedy_buffer = comparison['greedy']['avg_buffer']
        greedy_prob = comparison['greedy']['avg_link_prob']
        
        for method, name in zip(methods[1:], method_names[1:]):
            result = comparison[method]
            
            if greedy_buffer > 0:
                buffer_ratio = result['avg_buffer'] / greedy_buffer
                logger.info(f"  {name} Buffer: {buffer_ratio:.2f}x")
            
            if greedy_prob > 0:
                prob_ratio = result['avg_link_prob'] / greedy_prob
                logger.info(f"  {name} 링크 확률: {prob_ratio:.2f}x")
        
        logger.info("=" * 70)


def run_all_benchmarks(
    link_probs: np.ndarray,
    buffer_scores: np.ndarray,
    disrupted_nodes: List[int],
    tis_optimized_map: Dict[int, int],
    top_k: int = 100,
    min_prob: float = 0.1
) -> Dict:
    """
    모든 벤치마크 실행 (편의 함수)
    
    Parameters
    ----------
    link_probs : np.ndarray
    buffer_scores : np.ndarray
    disrupted_nodes : list
    tis_optimized_map : dict
    top_k : int
    min_prob : float
    
    Returns
    -------
    comparison : dict
        비교 결과
    """
    comparator = BenchmarkComparator(link_probs, buffer_scores)
    
    comparison = comparator.compare_all(
        disrupted_nodes,
        tis_optimized_map,
        top_k,
        min_prob
    )
    
    return comparison
