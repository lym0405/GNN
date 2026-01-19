"""
Temporal Graph Builder for SC-TGN
==================================
시계열 그래프 데이터 구축 (Track A용)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalGraphBuilder:
    """
    시계열 그래프 데이터 빌더
    
    각 연도별 네트워크를 시간순으로 정렬하여
    TGN 학습용 Temporal Event Stream 생성
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir).resolve()  # 절대 경로로 변환
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # 시계열 데이터
        self.years = [2020, 2021, 2022, 2023]
        self.networks = {}
        self.firm_to_idx = {}
        self.idx_to_firm = {}
    
    def build_temporal_data(
        self,
        train_ratio: float = 0.8
    ) -> Dict:
        """
        시계열 그래프 데이터 구축
        
        Returns
        -------
        temporal_data : Dict
            {
                'events': List of (timestamp, src, dst, edge_feat),
                'num_nodes': int,
                'train_mask': np.ndarray (boolean),
                'test_mask': np.ndarray (boolean),
                'node_features': torch.Tensor,
                'year_boundaries': Dict[year -> event_idx]
            }
        """
        logger.info("=" * 70)
        logger.info("🕐 시계열 그래프 데이터 구축 시작")
        logger.info("=" * 70)
        
        # 1. Firm ID 매핑 로드
        self._load_firm_mapping()
        
        # 2. 각 연도별 네트워크 로드
        events = []
        year_boundaries = {}
        
        for year in self.years:
            year_events = self._load_year_network(year)
            year_boundaries[year] = len(events)
            events.extend(year_events)
            logger.info(f"  ✓ {year}년: {len(year_events):,} 이벤트")
        
        total_events = len(events)
        logger.info(f"\n✅ 총 이벤트 수: {total_events:,}")
        
        # 3. Train/Test 분할 (시간 순서 유지!)
        # 마지막 연도(2023)를 Test로 사용
        train_boundary = year_boundaries[2023]
        train_mask = np.zeros(total_events, dtype=bool)
        train_mask[:train_boundary] = True
        test_mask = ~train_mask
        
        logger.info(f"  ✓ Train 이벤트: {train_mask.sum():,} (2020-2022)")
        logger.info(f"  ✓ Test 이벤트: {test_mask.sum():,} (2023)")
        
        # 4. 노드 피처 로드 (Phase 2 출력)
        node_features = self._load_node_features()
        
        # 5. 결과 반환
        temporal_data = {
            'events': events,
            'num_nodes': len(self.firm_to_idx),
            'train_mask': train_mask,
            'test_mask': test_mask,
            'node_features': node_features,
            'year_boundaries': year_boundaries
        }
        
        logger.info("=" * 70)
        return temporal_data
    
    def _load_firm_mapping(self):
        """기업 ID 매핑 로드"""
        mapping_path = self.raw_dir / "firm_to_idx_model2.csv"
        logger.info(f"📂 Firm mapping 경로: {mapping_path.absolute()}")
        
        if not mapping_path.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {mapping_path.absolute()}")
        
        df = pd.read_csv(mapping_path)
        
        # 컬럼명 확인 및 처리
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'firm_id'})
        
        # firm_id와 idx 컬럼 확인
        if 'firm_id' not in df.columns and len(df.columns) >= 2:
            # 첫 번째 컬럼을 firm_id로 사용
            df.columns = ['firm_id', 'idx'] + list(df.columns[2:])
        
        self.firm_to_idx = dict(zip(df['firm_id'], df['idx']))
        self.idx_to_firm = dict(zip(df['idx'], df['firm_id']))
        
        logger.info(f"✓ 기업 수: {len(self.firm_to_idx)}")
    
    def _load_year_network(self, year: int) -> List[Tuple]:
        """
        특정 연도의 네트워크 로드
        
        Returns
        -------
        events : List of (timestamp, src, dst, edge_feat)
        """
        # 실제 파일명: posco_network_capital_consumergoods_removed_{year}.csv
        network_path = self.raw_dir / f"posco_network_capital_consumergoods_removed_{year}.csv"
        
        # 폴백: 짧은 파일명도 시도
        if not network_path.exists():
            network_path = self.raw_dir / f"posco_network_{year}.csv"
        
        if not network_path.exists():
            logger.warning(f"⚠️  {year}년 네트워크 파일 없음")
            return []
        
        df = pd.read_csv(network_path)
        
        # 컬럼명 처리
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # source/target 컬럼명 확인
        # structure 문서 기준: 사업자등록번호, 거래처사업자등록번호
        source_col = None
        target_col = None
        amount_col = None
        
        # 1순위: 정확한 컬럼명 (실제 데이터)
        if '사업자등록번호' in df.columns:
            source_col = '사업자등록번호'
        if '거래처사업자등록번호' in df.columns:
            target_col = '거래처사업자등록번호'
        if '총공급금액' in df.columns:
            amount_col = '총공급금액'
        
        # 2순위: 부분 매칭
        if source_col is None:
            for col in df.columns:
                if '사업자' in col and '번호' in col and '거래처' not in col:
                    source_col = col
                    break
        
        if target_col is None:
            for col in df.columns:
                if '거래처' in col and '사업자' in col and '번호' in col:
                    target_col = col
                    break
        
        if amount_col is None:
            for col in df.columns:
                if '공급금액' in col or '거래액' in col or '금액' in col:
                    amount_col = col
                    break
        
        # 3순위: 영문 컬럼명 (더미 데이터용)
        if source_col is None or target_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if source_col is None and ('source' in col_lower or 'from' in col_lower):
                    source_col = col
                if target_col is None and ('target' in col_lower or 'to' in col_lower or 'dest' in col_lower):
                    target_col = col
                if amount_col is None and ('amount' in col_lower or 'weight' in col_lower):
                    amount_col = col
        
        # 컬럼이 없으면 첫 컬럼들을 사용
        if source_col is None or target_col is None:
            if len(df.columns) >= 2:
                source_col = df.columns[0]
                target_col = df.columns[1]
                if amount_col is None and len(df.columns) >= 3:
                    amount_col = df.columns[2]
                logger.info(f"  📋 {year}년 네트워크: '{source_col}' -> '{target_col}' (금액: '{amount_col}')")
            else:
                logger.error(f"❌ {year}년 네트워크 컬럼 부족")
                return []
        
        events = []
        base_timestamp = year * 365  # 연도를 일 단위로 변환
        
        for idx, row in df.iterrows():
            src_firm = row[source_col]
            dst_firm = row[target_col]
            
            # 기업 ID -> 인덱스 변환
            if src_firm not in self.firm_to_idx or dst_firm not in self.firm_to_idx:
                continue
            
            src_idx = self.firm_to_idx[src_firm]
            dst_idx = self.firm_to_idx[dst_firm]
            
            # 엣지 피처 (거래액, 빈도 등)
            edge_feat = self._extract_edge_features(row, amount_col)
            
            # 타임스탬프 (연도 내 순서)
            if 'timestamp' in df.columns:
                timestamp = base_timestamp + row['timestamp']
            else:
                timestamp = base_timestamp + idx  # 행 번호를 타임스탬프로
            
            events.append((timestamp, src_idx, dst_idx, edge_feat))
        
        # 시간순 정렬
        events.sort(key=lambda x: x[0])
        
        return events
    
    def _extract_edge_features(self, row: pd.Series, amount_col: str = None) -> np.ndarray:
        """
        엣지 피처 추출
        
        Parameters
        ----------
        row : pd.Series
            데이터프레임의 한 행
        amount_col : str
            거래액 컬럼명
        
        Returns
        -------
        edge_feat : np.ndarray [edge_dim]
        """
        features = []
        
        # 거래액 (정규화) - structure 문서 기준: 총공급금액
        if amount_col and amount_col in row:
            amount = row[amount_col]
            features.append(np.log1p(float(amount)) if pd.notna(amount) else 0.0)
        elif 'transaction_amount' in row:
            amount = row['transaction_amount']
            features.append(np.log1p(amount))
        else:
            features.append(0.0)
        
        # 거래 빈도
        if 'frequency' in row:
            features.append(row['frequency'])
        else:
            features.append(1.0)
        
        # 추가 피처들...
        
        return np.array(features, dtype=np.float32)
    
    def _load_node_features(self) -> torch.Tensor:
        """
        노드 피처 로드 (Phase 2 출력)
        
        Returns
        -------
        node_features : torch.Tensor [N, D]
        """
        feat_path = self.processed_dir / "X_feature_matrix.npy"
        
        if feat_path.exists():
            X = np.load(feat_path)
            logger.info(f"✓ 노드 피처 로드: {X.shape}")
            return torch.from_numpy(X).float()
        else:
            logger.warning("⚠️  노드 피처 없음, 영벡터 사용")
            return torch.zeros(len(self.firm_to_idx), 73)
    
    def get_temporal_edge_index(
        self,
        events: List[Tuple],
        mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        이벤트 리스트를 엣지 인덱스로 변환
        
        Returns
        -------
        edge_index : torch.Tensor [2, E]
        edge_attr : torch.Tensor [E, edge_dim]
        timestamps : torch.Tensor [E]
        """
        masked_events = [e for i, e in enumerate(events) if mask[i]]
        
        timestamps = torch.tensor([e[0] for e in masked_events], dtype=torch.long)
        edge_index = torch.tensor([[e[1], e[2]] for e in masked_events], dtype=torch.long).t()
        edge_attr = torch.tensor([e[3] for e in masked_events], dtype=torch.float32)
        
        return edge_index, edge_attr, timestamps
