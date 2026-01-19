"""
Curriculum Negative Sampler
============================
Easy → Hard 커리큘럼 학습을 위한 네거티브 샘플링

특징:
- Random Negative (Easy): 완전 랜덤
- Historical Hard Negative (Hard): 과거 거래 단절
- 점진적 Hard 비율 증가: 0% → 20% → 40% → 30%
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurriculumNegativeSampler:
    """
    커리큘럼 네거티브 샘플러
    
    Parameters
    ----------
    num_nodes : int
        전체 노드(기업) 수
    edge_index : torch.Tensor, shape (2, E)
        Positive 엣지 인덱스
    data_dir : str
        데이터 디렉토리 경로
    """
    
    def __init__(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        data_dir: str = "data"
    ):
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.data_dir = Path(data_dir)
        
        # Positive 엣지 집합
        self.pos_edge_set = set(
            map(tuple, edge_index.t().numpy())
        )
        
        # Historical Hard Negative 로드
        self.historical_negatives = self._load_historical_negatives()
        
        logger.info(f"✅ CurriculumNegativeSampler 초기화")
        logger.info(f"   - 노드 수: {num_nodes:,}")
        logger.info(f"   - Positive 엣지: {len(self.pos_edge_set):,}")
        logger.info(f"   - Historical Negatives: {len(self.historical_negatives):,}")
    
    def sample(
        self,
        num_samples: int,
        epoch: int,
        total_epochs: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        커리큘럼 샘플링
        
        Parameters
        ----------
        num_samples : int
            샘플링할 negative 엣지 수 (positive 대비 배수)
        epoch : int
            현재 에폭 (1부터 시작)
        total_epochs : int
            전체 에폭 수
        
        Returns
        -------
        neg_edge_index : torch.Tensor, shape (2, num_samples)
            Negative 엣지 인덱스
        neg_labels : torch.Tensor, shape (num_samples,)
            Negative 레이블 (모두 0)
        """
        # 커리큘럼 단계 결정
        hard_ratio = self._get_hard_ratio(epoch, total_epochs)
        
        num_hard = int(num_samples * hard_ratio)
        num_random = num_samples - num_hard
        
        logger.info(f"  📊 Epoch {epoch}: Random {num_random:,} / Hard {num_hard:,} ({hard_ratio*100:.0f}%)")
        
        # Random Negative 샘플링
        random_negs = self._sample_random_negatives(num_random)
        
        # Historical Hard Negative 샘플링
        if num_hard > 0 and len(self.historical_negatives) > 0:
            hard_negs = self._sample_hard_negatives(num_hard)
        else:
            hard_negs = np.array([]).reshape(0, 2)
        
        # 부족한 경우 Random으로 채우기
        if hard_negs.shape[0] < num_hard:
            shortage = num_hard - hard_negs.shape[0]
            extra_random = self._sample_random_negatives(shortage)
            random_negs = np.vstack([random_negs, extra_random])
        
        # 결합
        if hard_negs.shape[0] > 0:
            neg_edges = np.vstack([random_negs, hard_negs])
        else:
            neg_edges = random_negs
        
        # Shuffle
        np.random.shuffle(neg_edges)
        
        neg_edge_index = torch.from_numpy(neg_edges.T).long()
        neg_labels = torch.zeros(neg_edges.shape[0])
        
        return neg_edge_index, neg_labels
    
    def _get_hard_ratio(self, epoch: int, total_epochs: int) -> float:
        """
        커리큘럼에 따른 Hard Negative 비율 결정
        
        Epoch 1-20: 0% (Random only)
        Epoch 21-50: 20%
        Epoch 51-55: 40%
        Epoch 56-60: 30% (안정화)
        """
        if epoch <= 20:
            return 0.0  # Easy Phase (Random only)
        elif epoch <= 50:
            return 0.2  # Medium Phase
        elif epoch <= 55:
            return 0.4  # Hard Phase
        else:
            return 0.3  # Final Phase (안정화)
    
    def _sample_random_negatives(self, num_samples: int) -> np.ndarray:
        """
        벡터화된 랜덤 네거티브 샘플링 (속도 개선)
        
        최적화 전략:
        1. 한 번에 여러 개 생성하여 충돌 대비
        2. 벡터화된 연산으로 필터링
        3. Set 기반 중복 제거
        
        Returns
        -------
        neg_edges : np.ndarray, shape (num_samples, 2)
        """
        neg_edges = set()
        required = num_samples
        
        # 한 번에 1.5배수 정도 생성하여 충돌 대비
        multiplier = 1.5
        max_iterations = 100  # 무한 루프 방지
        iteration = 0
        
        while len(neg_edges) < required and iteration < max_iterations:
            iteration += 1
            curr_needed = required - len(neg_edges)
            n_gen = int(curr_needed * multiplier)
            
            # 벡터화된 난수 생성
            src = np.random.randint(0, self.num_nodes, size=n_gen)
            dst = np.random.randint(0, self.num_nodes, size=n_gen)
            
            # [최적화 1] Self-loop 제거 (벡터 연산)
            mask = src != dst
            src, dst = src[mask], dst[mask]
            
            # [최적화 2] Positive 및 중복 필터링
            for s, d in zip(src, dst):
                if (s, d) not in self.pos_edge_set:
                    neg_edges.add((s, d))
                    if len(neg_edges) >= required:
                        break
            
            # 루프가 너무 많이 돌지 않도록 multiplier 조정
            multiplier = min(multiplier * 1.2, 5.0)  # Cap at 5x
        
        # 만약 충분히 생성되지 않았다면 경고
        if len(neg_edges) < required:
            logger.warning(f"  ⚠️  Random negative 샘플링 부족: {len(neg_edges)}/{required}")
        
        # 리스트 변환 후 배열화 (필요한 만큼만)
        return np.array(list(neg_edges)[:num_samples])
    
    def _sample_hard_negatives(self, num_samples: int) -> np.ndarray:
        """
        Historical Hard Negative 샘플링
        
        Returns
        -------
        hard_negs : np.ndarray, shape (num_samples, 2)
        """
        if len(self.historical_negatives) == 0:
            return np.array([]).reshape(0, 2)
        
        # 중복 허용 샘플링
        indices = np.random.choice(
            len(self.historical_negatives),
            size=min(num_samples, len(self.historical_negatives)),
            replace=False
        )
        
        hard_negs = np.array([self.historical_negatives[i] for i in indices])
        
        return hard_negs
    
    def _load_historical_negatives(self) -> list:
        """
        2020-2023년 거래 엣지 로드 (2024년에는 없는 것들)
        
        Returns
        -------
        historical_negs : list of tuples
            [(src, dst), ...]
        """
        historical_negs = []
        
        # 2020-2023년 거래 데이터 로드
        for year in range(2020, 2024):
            csv_path = self.data_dir / "raw" / f"posco_network_capital_consumergoods_removed_{year}.csv"
            
            if not csv_path.exists():
                logger.warning(f"  ⚠️  Historical 데이터 없음: {year}")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                
                # 사업자번호 컬럼 찾기
                src_col = None
                dst_col = None
                
                for col in df.columns:
                    if '사업자등록번호' in col and '거래처' not in col:
                        src_col = col
                    elif '거래처' in col and '사업자등록번호' in col:
                        dst_col = col
                
                if src_col and dst_col:
                    # firm_to_idx 매핑 로드
                    firm_to_idx = self._load_firm_to_idx()
                    
                    for _, row in df.iterrows():
                        src_biz = str(row[src_col])
                        dst_biz = str(row[dst_col])
                        
                        if src_biz in firm_to_idx and dst_biz in firm_to_idx:
                            src_idx = firm_to_idx[src_biz]
                            dst_idx = firm_to_idx[dst_biz]
                            
                            # 2024년에는 없는 엣지만 추가
                            if (src_idx, dst_idx) not in self.pos_edge_set:
                                historical_negs.append((src_idx, dst_idx))
                
                logger.info(f"  ✓ {year}년 Historical Negatives: {len(historical_negs):,}")
            
            except Exception as e:
                logger.warning(f"  ⚠️  {year}년 데이터 로드 실패: {e}")
        
        # 중복 제거
        historical_negs = list(set(historical_negs))
        
        return historical_negs
    
    def _load_firm_to_idx(self) -> dict:
        """기업 ID → 인덱스 매핑 로드"""
        firm_to_idx_path = self.data_dir / "raw" / "firm_to_idx_model2.csv"
        df = pd.read_csv(firm_to_idx_path)
        
        firm_to_idx = {}
        for idx, row in df.iterrows():
            biz_no = str(row['사업자등록번호'])
            firm_to_idx[biz_no] = idx
        
        return firm_to_idx


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 70)
    print("CurriculumNegativeSampler 테스트")
    print("=" * 70)
    
    # 더미 데이터
    num_nodes = 1000
    num_edges = 5000
    
    # 랜덤 엣지 생성
    src = np.random.randint(0, num_nodes, size=num_edges)
    dst = np.random.randint(0, num_nodes, size=num_edges)
    edge_index = torch.from_numpy(np.vstack([src, dst])).long()
    
    # 샘플러 초기화
    sampler = CurriculumNegativeSampler(num_nodes, edge_index)
    
    # 각 에폭별 샘플링 테스트
    for epoch in [1, 20, 21, 50, 51, 55, 56, 60]:
        neg_edge_index, neg_labels = sampler.sample(
            num_samples=1000,
            epoch=epoch,
            total_epochs=60
        )
        
        print(f"\n✅ Epoch {epoch}")
        print(f"   - Negative 샘플 수: {neg_edge_index.shape[1]:,}")
        print(f"   - 레이블 shape: {neg_labels.shape}")
