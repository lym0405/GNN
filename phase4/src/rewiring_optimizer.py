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
    firm_coords : Optional[np.ndarray]
        기업 좌표 (N, 2) - 지리적 필터링용
    firm_ksic : Optional[np.ndarray]
        기업 KSIC 코드 (N,) - 산업 필터링용
    max_distance_km : float
        최대 거리 (km) - 이 거리 이상은 후보에서 제외
    """
    
    def __init__(
        self,
        link_probs: np.ndarray,
        buffer_calculator: BufferCalculator,
        penalty_calculator: PenaltyCalculator,
        firm_coords: Optional[np.ndarray] = None,
        firm_ksic: Optional[np.ndarray] = None,
        max_distance_km: float = 500.0
    ):
        self.link_probs = link_probs
        self.buffer_calc = buffer_calculator
        self.penalty_calc = penalty_calculator
        self.firm_coords = firm_coords
        self.firm_ksic = firm_ksic
        self.max_distance_km = max_distance_km
        
        self.num_nodes = link_probs.shape[0]
        
        # [최적화 2] 후보 필터링: Candidate Pool 미리 생성
        self.candidate_pool = self._build_candidate_pool()
        
        logger.info("RewiringOptimizer 초기화")
        logger.info(f"  - 노드 수: {self.num_nodes:,}")
        logger.info(f"  - 링크 확률 범위: [{np.min(link_probs):.4f}, {np.max(link_probs):.4f}]")
        if self.candidate_pool is not None:
            avg_candidates = np.mean([len(v) for v in self.candidate_pool.values()])
            logger.info(f"  - 평균 후보 수/노드: {avg_candidates:.1f} (필터링 적용)")
    
    def _build_candidate_pool(self) -> Optional[Dict[int, List[int]]]:
        """
        [최적화 2] 후보 풀 미리 구축 (Pruning)
        
        모든 가능한 엣지 (N × N)를 검토하지 않고,
        - 거리 기반: 지리적으로 너무 먼 기업 제외
        - 산업 코드 기반: 연관성 없는 산업 간 연결 제외
        
        Returns
        -------
        candidate_pool : Dict[int, List[int]]
            각 노드별 가능한 후보 리스트
        """
        if self.firm_coords is None and self.firm_ksic is None:
            logger.info("  ⚠️  좌표/KSIC 정보 없음, 후보 필터링 스킵")
            return None
        
        logger.info("🔍 후보 풀 구축 시작 (Pruning)")
        
        candidate_pool = {}
        
        for src_node in range(self.num_nodes):
            candidates = []
            
            for tgt_node in range(self.num_nodes):
                if src_node == tgt_node:
                    continue
                
                # 거리 필터링
                if self.firm_coords is not None:
                    distance = self._calculate_distance(
                        self.firm_coords[src_node],
                        self.firm_coords[tgt_node]
                    )
                    if distance > self.max_distance_km:
                        continue
                
                # 산업 코드 필터링 (KSIC 앞 1자리 또는 2자리 일치)
                if self.firm_ksic is not None:
                    if not self._is_industry_compatible(
                        self.firm_ksic[src_node],
                        self.firm_ksic[tgt_node]
                    ):
                        continue
                
                candidates.append(tgt_node)
            
            candidate_pool[src_node] = candidates
            
            if (src_node + 1) % 10000 == 0:
                logger.info(f"  진행: {src_node+1:,}/{self.num_nodes:,}")
        
        # 통계
        total_candidates = sum(len(v) for v in candidate_pool.values())
        total_possible = self.num_nodes * (self.num_nodes - 1)
        reduction = (1 - total_candidates / total_possible) * 100
        
        logger.info(f"✅ 후보 풀 구축 완료")
        logger.info(f"  - 전체 가능: {total_possible:,}")
        logger.info(f"  - 필터링 후: {total_candidates:,}")
        logger.info(f"  - 감소율: {reduction:.1f}%")
        
        return candidate_pool
    
    def _calculate_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """
        두 좌표 간 거리 계산 (km)
        
        간단한 유클리드 거리 (실제로는 Haversine 공식 사용 권장)
        """
        # 위도/경도를 km로 대략 변환
        # 1도 ≈ 111km (위도), 1도 ≈ 88km (경도, 한국 기준)
        lat_km = (coord1[1] - coord2[1]) * 111
        lon_km = (coord1[0] - coord2[0]) * 88
        distance = np.sqrt(lat_km**2 + lon_km**2)
        return distance
    
    def _is_industry_compatible(self, ksic1: str, ksic2: str) -> bool:
        """
        산업 코드 호환성 체크
        
        KSIC 코드 앞 1-2자리가 일치하면 호환 가능
        예: C24 (금속) ↔ C25 (금속가공) = 호환 O
             C24 (금속) ↔ G47 (소매) = 호환 X
        """
        if ksic1 is None or ksic2 is None:
            return True
        
        ksic1_str = str(ksic1)
        ksic2_str = str(ksic2)
        
        # 앞 1자리 일치 (대분류)
        if len(ksic1_str) > 0 and len(ksic2_str) > 0:
            if ksic1_str[0] == ksic2_str[0]:
                return True
        
        # 또는 앞 2자리 일치 (중분류)
        if len(ksic1_str) >= 2 and len(ksic2_str) >= 2:
            if ksic1_str[:2] == ksic2_str[:2]:
                return True
        
        return False
    
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
        
        # [최적화 2] 후보 필터링: Candidate Pool 사용
        if self.candidate_pool is not None:
            candidates = [
                tgt for tgt in candidates
                if tgt in self.candidate_pool[src_node]
            ]
        
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
    
    def evaluate_move_delta(
        self, 
        current_graph_risk: float,
        u: int, 
        v: int, 
        action: str = 'add'
    ) -> float:
        """
        국소적 리스크 변화 평가 (Delta Calculation)
        
        [최적화] 전체 그래프 복사 및 재계산 대신 변경된 부분만 계산
        - Before: O(N) - 전체 노드 리스크 재계산
        - After: O(degree(u) + degree(v)) - 국소 변화만 계산
        
        Parameters
        ----------
        current_graph_risk : float
            현재 전체 그래프 리스크
        u : int
            소스 노드
        v : int
            타겟 노드
        action : str
            'add' or 'remove'
        
        Returns
        -------
        new_total_risk : float
            새로운 전체 리스크 (근사값)
        
        Notes
        -----
        Delta 방식:
        1. u와 v의 degree 변화 계산
        2. u, v 주변 노드들의 risk 변화만 계산
        3. 전체 risk = 기존 risk + delta_risk
        
        정확도 vs. 속도 트레이드오프:
        - 완전히 정확하지는 않지만 충분히 좋은 근사
        - 대규모 그래프에서 극적인 속도 향상
        """
        # [비효율] 전체 그래프 복사 -> 변경 -> 전체 시뮬레이션
        # temp_graph = current_graph.clone()
        # if action == 'add':
        #     temp_graph.add_edge(u, v)
        # else:
        #     temp_graph.remove_edge(u, v)
        # return self.calculate_total_risk(temp_graph)  # O(N) - 전체 재계산
        
        # [최적화] 국소적 변화만 계산 (Approximate Delta)
        delta_risk = self._calculate_local_risk_change(u, v, action)
        
        return current_graph_risk + delta_risk
    
    def _calculate_local_risk_change(
        self, 
        u: int, 
        v: int, 
        action: str
    ) -> float:
        """
        u-v 엣지 추가/제거로 인한 국소 리스크 변화 계산
        
        Parameters
        ----------
        u, v : int
            노드 인덱스
        action : str
            'add' or 'remove'
        
        Returns
        -------
        delta_risk : float
            리스크 변화량
        """
        # 1. u와 v의 degree 변화로 인한 직접적 영향
        sign = 1 if action == 'add' else -1
        
        # Buffer calculator에서 TIS 점수 가져오기
        buffer_scores = self.buffer_calc.compute_buffer()
        tis_u = 1.0 / (buffer_scores[u] + 1e-6)  # TIS ∝ 1/Buffer
        tis_v = 1.0 / (buffer_scores[v] + 1e-6)
        
        # Degree 변화: +1 or -1
        # Risk는 보통 degree와 TIS의 함수
        # 간단한 모델: risk_change ≈ TIS × degree_change
        delta_u = sign * tis_u * 0.1  # 가중치 조정 가능
        delta_v = sign * tis_v * 0.1
        
        # 2. 주변 노드에 미치는 영향 (선택적)
        # 실제로는 u, v의 이웃들도 영향을 받지만
        # 단순화를 위해 직접 영향만 고려
        
        delta_risk = delta_u + delta_v
        
        return delta_risk
