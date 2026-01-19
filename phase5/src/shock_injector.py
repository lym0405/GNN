"""
Phase 5: 충격 시나리오 생성 (Shock Injection)
=============================================
공급망 네트워크에 충격을 주입하여 시뮬레이션
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ShockInjector:
    """
    공급망 충격 시나리오 생성
    
    Parameters
    ----------
    network_df : pd.DataFrame
        네트워크 데이터
    firm_features : Optional[np.ndarray]
        기업 features (Optional)
    """
    
    def __init__(
        self,
        network_df: pd.DataFrame,
        firm_features: Optional[np.ndarray] = None
    ):
        self.network_df = network_df
        self.firm_features = firm_features
        
        # 원본 데이터 백업
        self.original_network = network_df.copy()
        if firm_features is not None:
            self.original_features = firm_features.copy()
        
        logger.info(f"✅ ShockInjector 초기화")
        logger.info(f"   - 네트워크 엣지: {len(network_df):,}개")
        if firm_features is not None:
            logger.info(f"   - 기업 Features: {firm_features.shape}")
    
    def inject_edge_deletion(
        self,
        source_indices: List[int],
        target_indices: List[int],
        deletion_ratio: float = 1.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        엣지 삭제 충격 주입
        
        특정 공급자 → 수요자 간의 거래를 삭제
        
        Parameters
        ----------
        source_indices : List[int]
            공급자 노드 인덱스 리스트
        target_indices : List[int]
            수요자 노드 인덱스 리스트
        deletion_ratio : float
            삭제 비율 (0.0~1.0), 1.0 = 전부 삭제
        
        Returns
        -------
        shocked_network : pd.DataFrame
            충격 후 네트워크
        shock_info : Dict
            충격 정보
        """
        logger.info(f"🔨 엣지 삭제 충격 주입")
        logger.info(f"   - 공급자: {len(source_indices):,}개")
        logger.info(f"   - 수요자: {len(target_indices):,}개")
        logger.info(f"   - 삭제 비율: {deletion_ratio:.1%}")
        
        # 원본에서 시작
        shocked_network = self.original_network.copy()
        
        # 컬럼명 확인
        src_col, dst_col = self._get_edge_columns(shocked_network)
        
        # firm_to_idx 역매핑 필요 (인덱스 → firm_id)
        # 여기서는 간단히 인덱스를 직접 비교
        # 실제로는 firm_to_idx 역매핑 필요
        
        # 삭제 대상 찾기
        # 임시: firm_id를 인덱스로 가정
        delete_mask = (
            shocked_network[src_col].isin(source_indices) &
            shocked_network[dst_col].isin(target_indices)
        )
        
        deleted_edges = shocked_network[delete_mask]
        
        # 삭제 비율 적용
        if deletion_ratio < 1.0:
            n_to_delete = int(len(deleted_edges) * deletion_ratio)
            deleted_edges = deleted_edges.sample(n=n_to_delete, random_state=42)
        
        # 삭제 실행
        shocked_network = shocked_network[~shocked_network.index.isin(deleted_edges.index)]
        
        shock_info = {
            'type': 'edge_deletion',
            'deleted_edges_count': len(deleted_edges),
            'remaining_edges_count': len(shocked_network),
            'deletion_ratio': deletion_ratio,
            'source_indices': source_indices,
            'target_indices': target_indices,
            'deleted_edges': deleted_edges,
        }
        
        logger.info(f"   ✓ 삭제된 엣지: {len(deleted_edges):,}개")
        logger.info(f"   ✓ 남은 엣지: {len(shocked_network):,}개")
        
        return shocked_network, shock_info
    
    def inject_node_disruption(
        self,
        node_indices: List[int],
        disruption_ratio: float = 1.0,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        노드 기능 장애 충격 주입
        
        특정 기업의 생산/매출 능력을 0으로 설정
        
        Parameters
        ----------
        node_indices : List[int]
            충격 대상 노드 인덱스
        disruption_ratio : float
            장애 비율 (0.0~1.0)
        feature_columns : Optional[List[str]]
            영향받을 feature 컬럼 (None이면 전체)
        
        Returns
        -------
        shocked_features : np.ndarray
            충격 후 features
        shock_info : Dict
            충격 정보
        """
        if self.firm_features is None:
            logger.warning("⚠️  Features 없음, 충격 주입 불가")
            return None, {}
        
        logger.info(f"🔨 노드 기능 장애 충격 주입")
        logger.info(f"   - 대상 노드: {len(node_indices):,}개")
        logger.info(f"   - 장애 비율: {disruption_ratio:.1%}")
        
        # 원본에서 시작
        shocked_features = self.original_features.copy()
        
        # 장애 적용
        for node_idx in node_indices:
            if node_idx < len(shocked_features):
                if feature_columns is None:
                    # 전체 features를 disruption_ratio만큼 감소
                    shocked_features[node_idx] *= (1 - disruption_ratio)
                else:
                    # 특정 feature만 감소
                    # (실제로는 feature 이름 → 인덱스 매핑 필요)
                    pass
        
        shock_info = {
            'type': 'node_disruption',
            'affected_nodes_count': len(node_indices),
            'disruption_ratio': disruption_ratio,
            'node_indices': node_indices,
        }
        
        logger.info(f"   ✓ 노드 장애 적용 완료")
        
        return shocked_features, shock_info
    
    def inject_supply_cut(
        self,
        supplier_indices: List[int],
        buyer_indices: List[int],
        cut_ratio: float = 1.0
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Dict]:
        """
        공급 차단 충격 (엣지 삭제 + 노드 장애 복합)
        
        Parameters
        ----------
        supplier_indices : List[int]
            공급자 인덱스
        buyer_indices : List[int]
            수요자 인덱스
        cut_ratio : float
            차단 비율
        
        Returns
        -------
        shocked_network : pd.DataFrame
        shocked_features : Optional[np.ndarray]
        shock_info : Dict
        """
        logger.info(f"🔨 공급 차단 충격 주입 (복합)")
        
        # 1. 엣지 삭제
        shocked_network, edge_info = self.inject_edge_deletion(
            source_indices=supplier_indices,
            target_indices=buyer_indices,
            deletion_ratio=cut_ratio
        )
        
        # 2. 공급자 노드 장애
        shocked_features, node_info = self.inject_node_disruption(
            node_indices=supplier_indices,
            disruption_ratio=cut_ratio
        )
        
        shock_info = {
            'type': 'supply_cut',
            'edge_deletion': edge_info,
            'node_disruption': node_info,
            'cut_ratio': cut_ratio,
        }
        
        logger.info(f"   ✓ 복합 충격 주입 완료")
        
        return shocked_network, shocked_features, shock_info
    
    def _get_edge_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """네트워크 데이터의 source, target 컬럼 찾기"""
        if '사업자등록번호' in df.columns and '거래처사업자등록번호' in df.columns:
            return '사업자등록번호', '거래처사업자등록번호'
        elif 'source' in df.columns and 'target' in df.columns:
            return 'source', 'target'
        elif 'src' in df.columns and 'dst' in df.columns:
            return 'src', 'dst'
        else:
            logger.warning("⚠️  엣지 컬럼명 확인 필요")
            return df.columns[0], df.columns[1]
    
    def reset(self):
        """원본 데이터로 리셋"""
        self.network_df = self.original_network.copy()
        if self.firm_features is not None:
            self.firm_features = self.original_features.copy()
        
        logger.info("✅ 원본 데이터로 리셋 완료")


# ============================================================
# 유틸리티 함수
# ============================================================

def create_shock_scenario(
    network_df: pd.DataFrame,
    supplier_indices: List[int],
    buyer_indices: List[int],
    shock_type: str = 'edge_deletion',
    shock_intensity: float = 1.0,
    firm_features: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Dict]:
    """
    충격 시나리오 생성 (간편 함수)
    
    Parameters
    ----------
    network_df : pd.DataFrame
    supplier_indices : List[int]
    buyer_indices : List[int]
    shock_type : str
        'edge_deletion', 'node_disruption', 'supply_cut'
    shock_intensity : float
        충격 강도 (0.0~1.0)
    firm_features : Optional[np.ndarray]
    
    Returns
    -------
    shocked_network : pd.DataFrame
    shocked_features : Optional[np.ndarray]
    shock_info : Dict
    """
    injector = ShockInjector(network_df, firm_features)
    
    if shock_type == 'edge_deletion':
        shocked_network, shock_info = injector.inject_edge_deletion(
            source_indices=supplier_indices,
            target_indices=buyer_indices,
            deletion_ratio=shock_intensity
        )
        return shocked_network, firm_features, shock_info
    
    elif shock_type == 'node_disruption':
        shocked_features, shock_info = injector.inject_node_disruption(
            node_indices=supplier_indices,
            disruption_ratio=shock_intensity
        )
        return network_df, shocked_features, shock_info
    
    elif shock_type == 'supply_cut':
        return injector.inject_supply_cut(
            supplier_indices=supplier_indices,
            buyer_indices=buyer_indices,
            cut_ratio=shock_intensity
        )
    
    else:
        raise ValueError(f"Unknown shock_type: {shock_type}")


def propagate_shock_gpu(
    adj_matrix: torch.sparse.FloatTensor,
    initial_shock: torch.Tensor,
    steps: int = 30,
    activation_fn: str = 'sigmoid',
    threshold: float = 0.5
) -> torch.Tensor:
    """
    GPU 기반 병렬 충격 전파 시뮬레이션 (배치 처리)
    
    [최적화] 순차적 노드 반복 대신 행렬 곱으로 한 번에 처리
    - Before: for node in nodes: ... → O(N × steps)
    - After: Sparse Matrix Multiplication → O(nnz × steps)
    - 배치 처리: 100개 시나리오를 한 번의 연산으로 처리
    
    Parameters
    ----------
    adj_matrix : torch.sparse.FloatTensor
        인접 행렬 (N × N, sparse)
    initial_shock : torch.Tensor
        초기 충격 벡터
        - Shape: (batch_size, N) 또는 (N,)
        - 여러 시나리오 동시 실행 가능
    steps : int
        전파 스텝 수
    activation_fn : str
        활성화 함수 ('sigmoid', 'relu', 'threshold')
    threshold : float
        임계값 (activation_fn='threshold'일 때)
    
    Returns
    -------
    history : torch.Tensor
        충격 전파 히스토리
        - Shape: (steps, batch_size, N)
    
    Notes
    -----
    병렬 처리 예시:
    - 100개의 서로 다른 충격 시나리오를 동시에 시뮬레이션
    - GPU 메모리가 허용하는 한 무한정 확장 가능
    - 단일 시나리오 대비 100배 시나리오도 거의 같은 시간
    
    Example
    -------
    >>> adj = torch.sparse_coo_tensor(...)  # N × N
    >>> # 100개 시나리오
    >>> initial_shocks = torch.randn(100, N)  # 각 시나리오마다 다른 충격
    >>> history = propagate_shock_gpu(adj, initial_shocks, steps=30)
    >>> history.shape  # (30, 100, N)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # GPU로 이동
    current_status = initial_shock.to(device)
    adj = adj_matrix.to(device)
    
    # 배치 처리를 위한 차원 확인
    if current_status.dim() == 1:
        current_status = current_status.unsqueeze(0)  # (N,) → (1, N)
    
    batch_size, num_nodes = current_status.shape
    
    history = []
    
    logger.info(f"🚀 GPU 충격 전파 시작")
    logger.info(f"   - Device: {device}")
    logger.info(f"   - Batch Size: {batch_size}")
    logger.info(f"   - 노드 수: {num_nodes:,}")
    logger.info(f"   - Steps: {steps}")
    
    # [최적화 2] 조기 종료를 위한 변수
    convergence_threshold = 1e-4  # 변화량이 이 값 이하면 수렴으로 판단
    prev_status = None
    
    for step in range(steps):
        # [최적화] Sparse Matrix Multiplication
        # for node in nodes: ... 대신 행렬 곱 한 번
        # 100개의 시나리오를 한 번의 연산으로 처리 (Batch Processing)
        
        # current_status: (batch_size, N)
        # adj: (N, N) sparse
        # result: (batch_size, N)
        
        # torch.sparse.mm은 2D만 지원하므로 transpose 활용
        impact = torch.sparse.mm(adj, current_status.t()).t()
        
        # 활성화 함수 적용
        if activation_fn == 'sigmoid':
            current_status = torch.sigmoid(impact)
        elif activation_fn == 'relu':
            current_status = torch.relu(impact)
        elif activation_fn == 'threshold':
            current_status = (impact > threshold).float()
        else:
            current_status = impact  # identity
        
        history.append(current_status.clone())
        
        # [최적화 2] 조기 종료 (Early Stopping)
        # 더 이상 충격이 전파되지 않는 Steady State 도달 시 종료
        if prev_status is not None:
            # 모든 배치에 대해 변화량 계산
            diff = torch.abs(current_status - prev_status).max().item()
            
            if diff < convergence_threshold:
                logger.info(
                    f"   🛑 조기 종료 (Step {step+1}/{steps}): "
                    f"수렴 감지 (diff={diff:.6f})"
                )
                break
        
        prev_status = current_status.clone()
        
        if (step + 1) % 10 == 0:
            logger.info(f"   ✓ Step {step+1}/{steps}")
    
    # Stack: (steps, batch_size, N)
    history_tensor = torch.stack(history)
    
    logger.info(f"✅ GPU 충격 전파 완료")
    logger.info(f"   - 실제 Steps: {len(history)}/{steps}")
    logger.info(f"   - Output Shape: {history_tensor.shape}")
    
    return history_tensor


def propagate_shock_cpu(
    adj_matrix: np.ndarray,
    initial_shock: np.ndarray,
    steps: int = 30,
    convergence_threshold: float = 1e-4
) -> np.ndarray:
    """
    CPU 기반 충격 전파 (단일 시나리오)
    
    GPU가 없는 환경을 위한 폴백
    [최적화 2] 조기 종료 포함
    
    Parameters
    ----------
    adj_matrix : np.ndarray
        인접 행렬 (N × N)
    initial_shock : np.ndarray
        초기 충격 벡터 (N,)
    steps : int
        최대 전파 스텝 수
    convergence_threshold : float
        수렴 판단 임계값
    
    Returns
    -------
    history : np.ndarray
        충격 전파 히스토리 (actual_steps, N)
    """
    current_status = initial_shock.copy()
    history = []
    prev_status = None
    
    for step in range(steps):
        # 행렬 곱
        impact = adj_matrix @ current_status
        current_status = 1 / (1 + np.exp(-impact))  # sigmoid
        history.append(current_status.copy())
        
        # [최적화 2] 조기 종료
        if prev_status is not None:
            diff = np.abs(current_status - prev_status).max()
            if diff < convergence_threshold:
                logger.info(
                    f"   🛑 조기 종료 (Step {step+1}/{steps}): "
                    f"수렴 감지 (diff={diff:.6f})"
                )
                break
        
        prev_status = current_status.copy()
    
    return np.array(history)
