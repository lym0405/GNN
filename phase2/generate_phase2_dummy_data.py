"""
Phase 2 Dummy Data Generator
=============================
Phase 2 테스트를 위한 통합 더미 데이터 생성 (Phase 1 포함)

실행 방법:
    python generate_phase2_dummy_data.py --n_firms 1000
"""

import numpy as np
import pandas as pd
from scipy.sparse import random, save_npz
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_phase2_test_data(
    n_firms: int = 1000,
    density: float = 0.02,
    output_dir: str = "../data"
):
    """
    Phase 2 테스트를 위한 완전한 더미 데이터 생성
    
    포함:
    - Phase 1 출력 (disentangled_recipes.pkl)
    - 모든 원본 데이터 (H 행렬, 기업정보, 재무 등)
    """
    
    output_path = Path(output_dir)
    raw_path = output_path / "raw"
    processed_path = output_path / "processed"
    
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"🎲 Phase 2 테스트 데이터 생성 (N={n_firms})")
    logger.info("=" * 70)
    
    # ============================================================
    # 1. 원본 데이터 생성 (Phase 1용)
    # ============================================================
    
    logger.info("\n[1/3] 원본 데이터 생성...")
    
    # IO 테이블
    A_matrix = np.random.rand(33, 33).astype(np.float32)
    A_matrix = A_matrix / A_matrix.sum(axis=1, keepdims=True)
    sector_names = [f'sector_{i:02d}' for i in range(1, 34)]
    A_df = pd.DataFrame(A_matrix, index=sector_names, columns=sector_names)
    A_df.to_csv(raw_path / "A_33.csv")
    logger.info("   ✓ A_33.csv")
    
    # H 행렬 (거래 네트워크)
    H_matrix = random(n_firms, n_firms, density=density, format='csr', random_state=42)
    H_matrix = H_matrix * 1e9
    save_npz(raw_path / "H_csr_model2.npz", H_matrix)
    logger.info(f"   ✓ H_csr_model2.npz (밀도: {density*100:.2f}%)")
    
    # 기업 인덱스 매핑
    firm_ids = [f'firm_{i:06d}' for i in range(n_firms)]
    firm_to_idx = pd.DataFrame({
        '사업자등록번호': firm_ids,
        'idx': range(n_firms)
    })
    firm_to_idx.to_csv(raw_path / "firm_to_idx_model2.csv", index=False)
    logger.info("   ✓ firm_to_idx_model2.csv")
    
    # 기업 정보
    sector_codes = np.random.randint(1, 34, size=n_firms)
    latitudes = np.random.uniform(33, 43, size=n_firms)
    longitudes = np.random.uniform(124, 132, size=n_firms)
    
    firm_info = pd.DataFrame({
        '업체번호': [f'biz_{i:06d}' for i in range(n_firms)],
        '사업자등록번호': firm_ids,
        '산업코드': sector_codes,
        '위도': latitudes,
        '경도': longitudes,
        '기업명': [f'Company_{i}' for i in range(n_firms)]
    })
    firm_info.to_csv(
        raw_path / "vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv",
        index=False
    )
    logger.info("   ✓ 기업 정보 CSV")
    
    # 매출 데이터
    revenues = np.random.lognormal(mean=20, sigma=2, size=n_firms)
    revenue_df = pd.DataFrame({
        '업체번호': [f'biz_{i:06d}' for i in range(n_firms)],
        '사업자등록번호': firm_ids,
        'tg_2024_final': revenues
    })
    revenue_df.to_csv(raw_path / "tg_2024_filtered.csv", index=False)
    logger.info("   ✓ tg_2024_filtered.csv")
    
    # 수출액
    export_values = revenues * np.random.uniform(0, 0.5, size=n_firms)
    export_df = pd.DataFrame({
        '업체번호': [f'biz_{i:06d}' for i in range(n_firms)],
        'export_value': export_values
    })
    export_df.to_csv(raw_path / "export_estimation_value_final.csv", index=False)
    logger.info("   ✓ export_estimation_value_final.csv")
    
    # 자산
    assets = revenues * np.random.uniform(1, 5, size=n_firms)
    asset_df = pd.DataFrame({
        '업체번호': [f'biz_{i:06d}' for i in range(n_firms)],
        'asset': assets
    })
    asset_df.to_csv(raw_path / "asset_final_2024_6차.csv", index=False)
    logger.info("   ✓ asset_final_2024_6차.csv")
    
    # TIS 점수
    tis_scores = np.random.beta(2, 5, size=n_firms)
    tis_df = pd.DataFrame({
        '업체번호': [f'biz_{i:06d}' for i in range(n_firms)],
        'TIS': tis_scores
    })
    tis_df.to_csv(raw_path / "shock_after_P_v2.csv", index=False)
    logger.info("   ✓ shock_after_P_v2.csv")
    
    # Historical 데이터 (2020-2023) - Phase 3용 시계열 네트워크
    for year in range(2020, 2024):
        # 이전 년도 거래 데이터 (일부만 2024년과 겹침)
        hist_edges = int(n_firms * density * n_firms * 0.5)
        hist_src = np.random.choice(firm_ids, size=hist_edges)
        hist_dst = np.random.choice(firm_ids, size=hist_edges)
        hist_amount = np.random.exponential(1e8, size=hist_edges)
        
        hist_df = pd.DataFrame({
            'Unnamed: 0': hist_src,       # Phase 3에서 source로 사용
            'Unnamed: 1': hist_dst,        # Phase 3에서 target으로 사용
            '사업자등록번호': hist_src,
            '거래처사업자등록번호': hist_dst,
            '총공급금액': hist_amount
        })
        
        # Phase 3가 찾는 파일명으로 저장
        hist_path = raw_path / f"posco_network_{year}.csv"
        hist_df.to_csv(hist_path, index=False, encoding='utf-8-sig')
        logger.info(f"   ✓ posco_network_{year}.csv")
        
        # 기존 이름도 저장 (호환성)
        hist_path_old = raw_path / f"posco_network_capital_consumergoods_removed_{year}.csv"
        hist_df.to_csv(hist_path_old, index=False, encoding='utf-8-sig')
    
    # ============================================================
    # 2. Phase 1 출력 생성 (레시피)
    # ============================================================
    
    logger.info("\n[2/3] Phase 1 출력 생성 (레시피)...")
    
    # 간단한 레시피 생성 (산업별 패턴 반영)
    recipes = {}
    for i, firm_id in enumerate(firm_ids):
        sector = sector_codes[i] - 1  # 0-based
        
        # 해당 산업에 집중된 레시피 생성
        recipe = np.random.dirichlet(np.ones(33) * 0.5)
        
        # 자기 산업에 가중치 부여
        recipe[sector] *= 3
        recipe = recipe / recipe.sum()
        
        recipes[firm_id] = recipe.astype(np.float32)
    
    # 저장
    with open(processed_path / "disentangled_recipes.pkl", 'wb') as f:
        pickle.dump(recipes, f)
    logger.info(f"   ✓ disentangled_recipes.pkl ({len(recipes)} 기업)")
    
    # B 행렬도 저장 (선택적)
    B_matrix = np.array([recipes[fid] for fid in firm_ids])
    np.save(processed_path / "B_matrix.npy", B_matrix)
    logger.info("   ✓ B_matrix.npy")
    
    # ============================================================
    # 3. 통계 출력
    # ============================================================
    
    logger.info("\n[3/3] 데이터 통계...")
    logger.info(f"   - 기업 수: {n_firms:,}")
    logger.info(f"   - 엣지 수: {H_matrix.nnz:,}")
    logger.info(f"   - 밀도: {H_matrix.nnz / (n_firms ** 2) * 100:.4f}%")
    logger.info(f"   - 평균 매출: {revenues.mean() / 1e6:.2f}M 원")
    logger.info(f"   - 평균 TIS: {tis_scores.mean():.3f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Phase 2 테스트 데이터 생성 완료!")
    logger.info("=" * 70)
    logger.info(f"📁 출력 디렉토리:")
    logger.info(f"   - {raw_path.absolute()}")
    logger.info(f"   - {processed_path.absolute()}")
    logger.info("\n다음 명령어로 Phase 2 실행:")
    logger.info("    python main_phase2_fixed.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2 테스트 데이터 생성")
    parser.add_argument('--n_firms', type=int, default=1000, help='기업 수')
    parser.add_argument('--density', type=float, default=0.02, help='H 행렬 밀도')
    parser.add_argument('--output_dir', type=str, default='data', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    generate_phase2_test_data(
        n_firms=args.n_firms,
        density=args.density,
        output_dir=args.output_dir
    )
