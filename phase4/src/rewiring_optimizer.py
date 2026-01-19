"""
Rewiring Optimizer: 제약 기반 최적 재배선 알고리즘
==================================================

Score_final = P(u,v) × Buffer(v) - Penalty_inv(u,v)

최고 점수를 가진 재배선 선택
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional
import logging

from .buffer_calculator import BufferCalculator
from .penalty_calculator import PenaltyCalculator

logger = logging.getLogger(__name__)


class RewiringOptimizer:
    """
    TIS-Optimized 재배선 최적화기
    
    Parameters
    ----------
    link_probs : np.ndarray [N, N]
        링크 예측 확률 (Phase 3)
    buffer_calculator : BufferCalculator
        충격완충력 계산기
    penalty_calculator : PenaltyCalculator
        패널티 계산기
    """
    
    def __init__(
        self,
        link_probs: np.ndarray,
        buffer_calculator: BufferCalculator,
        penalty_calculator: PenaltyCalculator
    ):
        self.link_probs = link_probs
        self.buffer_calc = buffer_calculator
        self.penalty_calc = penalty_calculator
        
        self.num_nodes = link_probs.shape[0]
        
        logger.info("RewiringOptimizer 초기화")
        logger.info(f"  - 노드 수: {self.num_nodes:,}")
        logger.info(f"  - 링크 확률 범위: [{np.min(link_probs):.4f}, {np.max(link_probs):.4f}]")
    
    def optimize_rewiring(
        self,
        disrupted_nodes: List[int],
        top_k: int = 100,
        min_prob_threshold: float = 0.1,
        min_buffer_threshold: float = 0.1,
        max_recipe_distance: float = 0.8
    ) -> Dict[int, int]:
        """
        제약 기반 최적 재배선
        
        Parameters
        ----------
        disrupted_nodes : List[int]
            단절 대상 노드 리스트
        top_k : int
            후보군 크기
        min_prob_threshold : float
            최소 링크 확률
        min_buffer_threshold : float
            최소 충격완충력
        max_recipe_distance : float
            최대 레시피 거리
        
        Returns
        -------
        rewiring_map : Dict[int, int]
            소스 → 타겟 재배선 매핑
        """
        logger.info("=" * 70)
        logger.info("제약 기반 최적 재배선 시작")
        logger.info("=" * 70)
        logger.info(f"단절 노드 수: {len(disrupted_nodes):,}")
        logger.info(f"후보군 크기: {top_k}")
        logger.info(f"최소 확률: {min_prob_threshold}")
        logger.info(f"최소 Buffer: {min_buffer_threshold}")
        logger.info(f"최대 레시피 거리: {max_recipe_distance}")
        
        rewiring_map = {}
        
        # Buffer 미리 계산
        buffer_scores = self.buffer_calc.compute_buffer()
        
        for i, src_node in enumerate(disrupted_nodes):
            if (i + 1) % 100 == 0:
                logger.info(f"  진행: {i+1}/{len(disrupted_nodes)}")
            
            # Step 1: 후보군 선정
            candidates = self._select_candidates(
                src_node,
                top_k,
                min_prob_threshold
            )
            
            if len(candidates) == 0:
                logger.warning(f"  ⚠️  노드 {src_node}: 후보군 없음")
                continue
            
            # Step 2: 최종 스코어 계산
            best_target, best_score = self._compute_best_target(
                src_node,
                candidates,
                buffer_scores,
                min_buffer_threshold,
                max_recipe_distance
            )
            
            if best_target is not None:
                rewiring_map[src_node] = best_target
            else:
                logger.warning(f"  ⚠️  노드 {src_node}: 적합한 타겟 없음")
        
        logger.info("=" * 70)
        logger.info(f"✅ 재배선 완료: {len(rewiring_map)}/{len(disrupted_nodes)}")
        logger.info(f"   성공률: {len(rewiring_map)/len(disrupted_nodes)*100:.1f}%")
        logger.info("=" * 70)
        
        return rewiring_map
    
    def _select_candidates(
        self,
        src_node: int,
        top_k: int,
        min_prob_threshold: float
    ) -> np.ndarray:
        """
        후보군 선정
        
        Parameters
        ----------
        src_node : int
            소스 노드
        top_k : int
            상위 K개
        min_prob_threshold : float
            최소 확률
        
        Returns
        -------
        candidates : np.ndarray
            후보 노드 인덱스
        """
        # 링크 확률
        probs = self.link_probs[src_node]
        
        # 최소 임계값 이상
        valid_mask = probs >= min_prob_threshold
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return np.array([], dtype=int)
        
        # 상위 K개 선택
        valid_probs = probs[valid_indices]
        top_k_local = min(top_k, len(valid_indices))
        top_k_indices = np.argpartition(valid_probs, -top_k_local)[-top_k_local:]
        
        candidates = valid_indices[top_k_indices]
        
        return candidates
    
    def _compute_best_target(
        self,
        src_node: int,
        candidates: np.ndarray,
        buffer_scores: np.ndarray,
        min_buffer_threshold: float,
        max_recipe_distance: float
    ) -> Tuple[Optional[int], float]:
        """
        최고 점수 타겟 선택
        
        Parameters
        ----------
        src_node : int
            소스 노드
        candidates : np.ndarray
            후보 노드들
        buffer_scores : np.ndarray
            충격완충력 점수
        min_buffer_threshold : float
            최소 Buffer
        max_recipe_distance : float
            최대 레시피 거리
        
        Returns
        -------
        best_target : int or None
            최고 점수 타겟
        best_score : float
            최고 점수
        """
        best_target = None
        best_score = -np.inf
        
        for tgt in candidates:
            # 제약 조건 체크
            buffer = buffer_scores[tgt]
            if buffer < min_buffer_threshold:
                continue
            
            # 레시피 불일치 체크
            recipe_mismatch = self.penalty_calc.compute_recipe_mismatch(
                src_node, tgt
            )
            if recipe_mismatch > max_recipe_distance:
                continue
            
            # 최종 스코어 계산
            score = self._compute_final_score(
                src_node, tgt, buffer_scores[tgt]
            )
            
            if score > best_score:
                best_score = score
                best_target = tgt
        
        return best_target, best_score
    
    def _compute_final_score(
        self,
        src_node: int,
        tgt_node: int,
        buffer: float
    ) -> float:
        """
        최종 스코어 계산
        
        Score = P(u,v) × Buffer(v) - Penalty_inv(u,v)
        
        Parameters
        ----------
        src_node : int
        tgt_node : int
        buffer : float
        
        Returns
        -------
        score : float
        """
        # 링크 확률
        prob = self.link_probs[src_node, tgt_node]
        
        # 패널티
        penalty = self.penalty_calc.compute_penalty(src_node, tgt_node)
        
        # 최종 스코어
        score = prob * buffer - penalty
        
        return score
    
    def create_rewired_network(
        self,
        H_original: sp.csr_matrix,
        rewiring_map: Dict[int, int]
    ) -> sp.csr_matrix:
        """
        재배선된 네트워크 생성
        
        Parameters
        ----------
        H_original : sp.csr_matrix [N, N]
            원본 네트워크
        rewiring_map : Dict[int, int]
            재배선 매핑
        
        Returns
        -------
        H_rewired : sp.csr_matrix [N, N]
            재배선된 네트워크
        """
        logger.info("재배선 네트워크 생성")
        
        # 원본 복사
        H_rewired = H_original.copy()
        
        # 재배선 적용
        for src, new_tgt in rewiring_map.items():
            # 기존 연결 삭제 (src의 모든 out-edges)
            H_rewired[src, :] = 0
            
            # 새 연결 추가
            # 엣지 가중치는 원본의 평균값 사용
            avg_weight = H_original[src, :].mean() if H_original[src, :].nnz > 0 else 1.0
            H_rewired[src, new_tgt] = avg_weight
        
        # 희소 행렬 정리
        H_rewired.eliminate_zeros()
        
        logger.info(f"✅ 재배선 네트워크 생성 완료")
        logger.info(f"   - 원본 엣지: {H_original.nnz:,}")
        logger.info(f"   - 재배선 엣지: {H_rewired.nnz:,}")
        logger.info(f"   - 변경된 엣지: {len(rewiring_map):,}")
        
        return H_rewired
    
    def evaluate_rewiring(
        self,
        rewiring_map: Dict[int, int],
        buffer_scores: np.ndarray
    ) -> Dict:
        """
        재배선 품질 평가
        
        Parameters
        ----------
        rewiring_map : Dict[int, int]
        buffer_scores : np.ndarray
        
        Returns
        -------
        metrics : Dict
            평가 지표
        """
        if len(rewiring_map) == 0:
            return {}
        
        targets = list(rewiring_map.values())
        
        # 평균 Buffer
        avg_buffer = np.mean(buffer_scores[targets])
        
        # 평균 TIS (Buffer와 역관계)
        tis_scores = self.buffer_calc.tis_scores
        avg_tis = np.mean(tis_scores[targets])
        
        # 레시피 유사도
        recipe_sims = []
        for src, tgt in rewiring_map.items():
            mismatch = self.penalty_calc.compute_recipe_mismatch(src, tgt)
            recipe_sims.append(1.0 - mismatch)
        avg_recipe_sim = np.mean(recipe_sims)
        
        metrics = {
            'num_rewired': len(rewiring_map),
            'avg_buffer': avg_buffer,
            'avg_tis': avg_tis,
            'avg_recipe_similarity': avg_recipe_sim
        }
        
        logger.info("재배선 품질 평가:")
        logger.info(f"  - 재배선 수: {metrics['num_rewired']:,}")
        logger.info(f"  - 평균 Buffer: {metrics['avg_buffer']:.4f}")
        logger.info(f"  - 평균 TIS: {metrics['avg_tis']:.4f}")
        logger.info(f"  - 평균 레시피 유사도: {metrics['avg_recipe_similarity']:.4f}")
        
        return metrics
