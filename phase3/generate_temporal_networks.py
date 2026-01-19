"""
Phase 3용 더미 시계열 네트워크 생성
====================================
posco_network_2020~2023.csv 생성
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_temporal_networks(
    n_firms: int = 500,
    density: float = 0.03,
    output_dir: str = "data/raw"
):
    """
    시계열 네트워크 더미 데이터 생성
    
    각 연도별로 약간씩 다른 네트워크 생성 (현실적인 변화 반영)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("🕐 Phase 3 시계열 네트워크 더미 데이터 생성")
    logger.info("=" * 70)
    
    # Firm IDs (firm_to_idx 활용)
    firm_to_idx_path = output_path / "firm_to_idx_model2.csv"
    if firm_to_idx_path.exists():
        df_firms = pd.read_csv(firm_to_idx_path)
        if 'Unnamed: 0' in df_firms.columns:
            firm_ids = df_firms['Unnamed: 0'].values
        else:
            firm_ids = np.arange(n_firms)
    else:
        firm_ids = np.arange(n_firms)
    
    n_firms = len(firm_ids)
    base_edges_per_year = int(n_firms * n_firms * density)
    
    logger.info(f"   - 기업 수: {n_firms}")
    logger.info(f"   - 밀도: {density}")
    logger.info(f"   - 연도당 기본 엣지 수: {base_edges_per_year:,}")
    
    # 연도별 네트워크 생성
    years = [2020, 2021, 2022, 2023]
    all_edges = set()
    
    for year in years:
        # 약간의 변화 (±10%)
        n_edges = int(base_edges_per_year * np.random.uniform(0.9, 1.1))
        
        edges = []
        year_edges = set()
        
        # 기존 엣지의 80%를 유지 (연속성)
        if len(all_edges) > 0:
            retained = list(all_edges)
            np.random.shuffle(retained)
            keep_count = int(len(retained) * 0.8)
            edges.extend(retained[:keep_count])
            year_edges.update(retained[:keep_count])
        
        # 새로운 엣지 추가
        attempts = 0
        max_attempts = n_edges * 5
        
        while len(edges) < n_edges and attempts < max_attempts:
            src = np.random.choice(firm_ids)
            dst = np.random.choice(firm_ids)
            
            if src != dst and (src, dst) not in year_edges:
                edges.append((src, dst))
                year_edges.add((src, dst))
            
            attempts += 1
        
        # DataFrame 생성
        df = pd.DataFrame(edges, columns=['source', 'target'])
        
        # 엣지 피처 추가
        df['transaction_amount'] = np.random.lognormal(10, 2, len(df))  # 거래액
        df['frequency'] = np.random.randint(1, 20, len(df))  # 거래 빈도
        
        # 저장
        output_file = output_path / f"posco_network_{year}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"   ✓ {year}년: {len(df):,} 엣지 → {output_file.name}")
        
        # 다음 연도를 위해 저장
        all_edges = year_edges.copy()
    
    logger.info("=" * 70)
    logger.info("✅ 시계열 네트워크 더미 데이터 생성 완료!")
    logger.info("=" * 70)


if __name__ == "__main__":
    generate_temporal_networks(
        n_firms=500,
        density=0.03,
        output_dir="data/raw"
    )
