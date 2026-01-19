"""
Phase 3: Negative Sampling for Link Prediction
===============================================
Random + Historical Negative Sampling

네거티브 샘플링 전략:
    1. Random Negative: 무작위 기업쌍 (기초 학습)
    2. Historical Negative: 과거 존재했던 거래 기업쌍 (디테일 학습)
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple, Set, List
import logging

logger = logging.getLogger(__name__)


class Phase3NegativeSampler:
    """
    Phase 3용 네거티브 샘플러 (Random + Historical)
    
    Parameters
    ----------
    num_nodes : int
        전체 노드 수
    current_edges : torch.Tensor [2, E]
        현재 존재하는 엣지 (Positive)
    data_dir : str
        데이터 디렉토리 (historical negatives 로드용)
    """
    
    def __init__(
        self,
        num_nodes: int,
        current_edges: torch.Tensor,
        data_dir: str = "data"
    ):
        self.num_nodes = num_nodes
        self.current_edges = current_edges
        self.data_dir = Path(data_dir)
        
        # Positive 엣지를 set으로 저장 (빠른 검색)
        self.positive_set = set(
            map(tuple, current_edges.t().numpy())
        )
        
        # Historical Negatives 로드
        self.historical_negatives = self._load_historical_negatives()
        
        logger.info(f"✅ Phase3NegativeSampler 초기화")
        logger.info(f"   - 노드 수: {num_nodes}")
        logger.info(f"   - Positive 엣지: {len(self.positive_set):,}")
        logger.info(f"   - Historical Negatives: {len(self.historical_negatives):,}")
    
    def _load_historical_negatives(self) -> Set[Tuple[int, int]]:
        """
        과거 연도의 엣지를 Historical Negatives로 로드
        
        Returns
        -------
        historical_negatives : Set[Tuple[int, int]]
        """
        historical_set = set()
        
        # 과거 네트워크 파일들 (2020-2023)
        network_files = [
            self.data_dir / "raw" / f"posco_network_{year}.csv"
            for year in [2020, 2021, 2022, 2023]
        ]
        
        # firm_to_idx 로드
        firm_to_idx_path = self.data_dir / "raw" / "firm_to_idx_model2.csv"
        if not firm_to_idx_path.exists():
            logger.warning("⚠️  firm_to_idx 파일 없음, Historical Negatives 사용 불가")
            return historical_set
        
        firm_to_idx_df = pd.read_csv(firm_to_idx_path)
        
        # 'Unnamed: 0', 'idx' 처리
        if 'Unnamed: 0' in firm_to_idx_df.columns:
            firm_to_idx = dict(zip(
                firm_to_idx_df['Unnamed: 0'],
                firm_to_idx_df['idx']
            ))
        elif 'firm_id' in firm_to_idx_df.columns and 'idx' in firm_to_idx_df.columns:
            firm_to_idx = dict(zip(
                firm_to_idx_df['firm_id'],
                firm_to_idx_df['idx']
            ))
        else:
            logger.warning("⚠️  firm_to_idx 컬럼명 불일치")
            return historical_set
        
        # 각 연도별 네트워크 로드
        for year_idx, file_path in enumerate(network_files):
            if not file_path.exists():
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # 컬럼명 확인 (structure 문서 기준)
                src_col = None
                dst_col = None
                
                # 1순위: 한글 컬럼명 (실제 데이터)
                if '사업자등록번호' in df.columns:
                    src_col = '사업자등록번호'
                if '거래처사업자등록번호' in df.columns:
                    dst_col = '거래처사업자등록번호'
                
                # 2순위: 영문 컬럼명 (더미 데이터)
                if src_col is None and dst_col is None:
                    if 'Unnamed: 0' in df.columns and 'Unnamed: 1' in df.columns:
                        src_col, dst_col = 'Unnamed: 0', 'Unnamed: 1'
                    elif 'source' in df.columns and 'target' in df.columns:
                        src_col, dst_col = 'source', 'target'
                    elif 'src' in df.columns and 'dst' in df.columns:
                        src_col, dst_col = 'src', 'dst'
                
                if src_col is None or dst_col is None:
                    logger.warning(f"⚠️  {file_path.name}: 컬럼명 불일치")
                    continue
                
                # 인덱스 변환
                for _, row in df.iterrows():
                    src_firm = row[src_col]
                    dst_firm = row[dst_col]
                    
                    if src_firm in firm_to_idx and dst_firm in firm_to_idx:
                        src_idx = firm_to_idx[src_firm]
                        dst_idx = firm_to_idx[dst_firm]
                        
                        # 현재 positive가 아닌 것만 추가
                        if (src_idx, dst_idx) not in self.positive_set:
                            historical_set.add((src_idx, dst_idx))
                
                logger.info(f"   ✓ {2020 + year_idx}년: {len(historical_set):,}개 추가")
            
            except Exception as e:
                logger.warning(f"⚠️  {file_path.name} 로드 실패: {e}")
        
        return historical_set
    
    def sample_negatives(
        self,
        num_samples: int,
        historical_ratio: float = 0.5,
        seed: int = 42
    ) -> torch.Tensor:
        """
        네거티브 엣지 샘플링 (Random + Historical)
        
        Parameters
        ----------
        num_samples : int
            샘플링할 네거티브 개수
        historical_ratio : float
            Historical Negatives 비율 (0.0~1.0)
        seed : int
        
        Returns
        -------
        negative_edges : torch.Tensor [2, num_samples]
        """
        np.random.seed(seed)
        
        num_historical = int(num_samples * historical_ratio)
        num_random = num_samples - num_historical
        
        negatives = []
        
        # 1. Historical Negatives
        if num_historical > 0 and len(self.historical_negatives) > 0:
            historical_list = list(self.historical_negatives)
            if len(historical_list) >= num_historical:
                sampled = np.random.choice(
                    len(historical_list),
                    size=num_historical,
                    replace=False
                )
                negatives.extend([historical_list[i] for i in sampled])
            else:
                # 부족하면 전부 사용
                negatives.extend(historical_list)
                num_random += (num_historical - len(historical_list))
        else:
            # Historical 없으면 Random으로 대체
            num_random += num_historical
        
        # 2. Random Negatives
        if num_random > 0:
            attempts = 0
            max_attempts = num_random * 10
            
            while len(negatives) < num_samples and attempts < max_attempts:
                src = np.random.randint(0, self.num_nodes)
                dst = np.random.randint(0, self.num_nodes)
                
                # Self-loop 제거 & Positive 제거 & 중복 제거
                if (src != dst and 
                    (src, dst) not in self.positive_set and
                    (src, dst) not in negatives):
                    negatives.append((src, dst))
                
                attempts += 1
        
        # Tensor 변환
        if len(negatives) < num_samples:
            logger.warning(
                f"⚠️  요청된 {num_samples}개 중 {len(negatives)}개만 샘플링됨"
            )
        
        negative_edges = torch.tensor(
            negatives[:num_samples],
            dtype=torch.long
        ).t()  # [2, N]
        
        return negative_edges
    
    def sample_for_events(
        self,
        events: List[tuple],
        historical_ratio: float = 0.5,
        neg_ratio: float = 1.0,
        seed: int = 42
    ) -> List[tuple]:
        """
        이벤트 리스트에 대해 네거티브 샘플링
        
        Parameters
        ----------
        events : List[tuple]
            [(timestamp, src, dst, edge_feat), ...]
        historical_ratio : float
            Historical Negatives 비율
        neg_ratio : float
            Positive 1개당 Negative 개수
        seed : int
        
        Returns
        -------
        augmented_events : List[tuple]
            [(timestamp, src, dst, edge_feat, label), ...]
        """
        # 이벤트가 0개인 경우 처리
        if len(events) == 0:
            logger.warning("⚠️  이벤트가 0개입니다. 빈 리스트 반환")
            return []
        
        # Positive에 label=1.0 추가
        augmented = [(e[0], e[1], e[2], e[3], 1.0) for e in events]
        
        # 네거티브 샘플링
        num_negatives = int(len(events) * neg_ratio)
        negative_edges = self.sample_negatives(
            num_samples=num_negatives,
            historical_ratio=historical_ratio,
            seed=seed
        )
        
        # negative_edges가 비어있는 경우 처리
        if negative_edges.shape[1] == 0:
            logger.warning("⚠️  네거티브 샘플링 실패, Positive만 반환")
            return augmented
        
        # 네거티브 이벤트 생성 (랜덤 timestamp, zero edge_feat)
        np.random.seed(seed)
        for i in range(negative_edges.shape[1]):
            src = negative_edges[0, i].item()
            dst = negative_edges[1, i].item()
            
            # 랜덤 타임스탬프 (기존 이벤트 중에서)
            random_event = events[np.random.randint(len(events))]
            timestamp = random_event[0]
            edge_feat = np.zeros_like(random_event[3])
            
            augmented.append((timestamp, src, dst, edge_feat, 0.0))
        
        # 시간순 정렬
        augmented.sort(key=lambda x: x[0])
        
        logger.info(f"   ✓ Positive: {len(events):,}, Negative: {negative_edges.shape[1]:,}")
        
        return augmented


# ============================================================
# 유틸리티 함수
# ============================================================

def prepare_events_with_negatives(
    events: List[tuple],
    mask: np.ndarray,
    num_nodes: int,
    current_edges: torch.Tensor,
    data_dir: str = "data",
    historical_ratio: float = 0.5,
    neg_ratio: float = 1.0,
    seed: int = 42
) -> List[tuple]:
    """
    이벤트에 네거티브 샘플링 적용 (Random + Historical)
    
    Parameters
    ----------
    events : List[tuple]
        전체 이벤트 리스트
    mask : np.ndarray
        사용할 이벤트 마스크
    num_nodes : int
    current_edges : torch.Tensor
        현재 positive 엣지
    data_dir : str
    historical_ratio : float
    neg_ratio : float
    seed : int
    
    Returns
    -------
    augmented_events : List[tuple]
    """
    # 마스크된 이벤트만 선택
    selected_events = [e for i, e in enumerate(events) if mask[i]]
    
    logger.info(f"🎲 네거티브 샘플링 (Historical {historical_ratio*100:.0f}%, Neg Ratio {neg_ratio})")
    logger.info(f"   ✓ Positive 이벤트: {len(selected_events):,}")
    
    # 샘플러 초기화
    sampler = Phase3NegativeSampler(
        num_nodes=num_nodes,
        current_edges=current_edges,
        data_dir=data_dir
    )
    
    # 네거티브 샘플링
    augmented_events = sampler.sample_for_events(
        events=selected_events,
        historical_ratio=historical_ratio,
        neg_ratio=neg_ratio,
        seed=seed
    )
    
    logger.info(f"   ✓ 총 이벤트: {len(augmented_events):,}")
    
    return augmented_events
