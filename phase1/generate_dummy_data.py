"""
Dummy Data Generator for Phase 1 Testing
==========================================
Phase 1을 테스트하기 위한 더미 데이터 생성

실제 데이터가 없을 때 파이프라인 테스트용으로 사용
"""

import numpy as np
import pandas as pd
from scipy.sparse import random, save_npz
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dummy_data(
    n_firms: int = 10000,
    density: float = 0.01,
    output_dir: str = "../data/raw"
):
    """
    더미 데이터 생성
    
    Parameters
    ----------
    n_firms : int
        기업 수
    density : float
        H 행렬 밀도
    output_dir : str
        출력 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎲 더미 데이터 생성 시작 (N={n_firms})")
    
    # 1. IO 테이블 (33×33)
    logger.info("1️⃣ IO 테이블 생성...")
    A_matrix = np.random.rand(33, 33).astype(np.float32)
    A_matrix = A_matrix / A_matrix.sum(axis=1, keepdims=True)  # 행 합 = 1
    
    sector_names = [f'sector_{i:02d}' for i in range(1, 34)]
    A_df = pd.DataFrame(A_matrix, index=sector_names, columns=sector_names)
    A_df.to_csv(output_path / "A_33.csv")
    logger.info(f"   ✓ A_33.csv 저장")
    
    # 2. H 행렬 (거래 네트워크, Sparse)
    logger.info("2️⃣ H 행렬 생성...")
    H_matrix = random(n_firms, n_firms, density=density, format='csr', random_state=42)
    H_matrix = H_matrix * 1e9  # 거래 금액 (원)
    save_npz(output_path / "H_csr_model2.npz", H_matrix)
    logger.info(f"   ✓ H_csr_model2.npz 저장 (밀도: {density*100:.2f}%)")
    
    # 3. 기업 인덱스 매핑
    logger.info("3️⃣ 기업 인덱스 매핑 생성...")
    firm_ids = [f'firm_{i:06d}' for i in range(n_firms)]
    firm_to_idx = pd.DataFrame({
        '사업자등록번호': firm_ids,
        'idx': range(n_firms)
    })
    firm_to_idx.to_csv(output_path / "firm_to_idx_model2.csv", index=False)
    logger.info(f"   ✓ firm_to_idx_model2.csv 저장")
    
    # 4. 기업 정보 (산업코드 포함)
    logger.info("4️⃣ 기업 정보 생성...")
    
    # 산업코드: 1~33 랜덤 할당
    sector_codes = np.random.randint(1, 34, size=n_firms)
    
    # 좌표: 대한민국 범위 (위도 33~43, 경도 124~132)
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
        output_path / "vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv",
        index=False
    )
    logger.info(f"   ✓ 기업 정보 CSV 저장")
    
    # 5. 매출 데이터
    logger.info("5️⃣ 매출 데이터 생성...")
    
    # 로그정규분포로 매출 생성 (현실적)
    revenues = np.random.lognormal(mean=20, sigma=2, size=n_firms)  # 천원 단위
    
    revenue_df = pd.DataFrame({
        '업체번호': [f'biz_{i:06d}' for i in range(n_firms)],
        '사업자등록번호': firm_ids,
        'tg_2024_final': revenues
    })
    
    revenue_df.to_csv(output_path / "tg_2024_filtered.csv", index=False)
    logger.info(f"   ✓ tg_2024_filtered.csv 저장")
    
    # 매출 추정 (일부만)
    revenue_est = revenue_df.sample(frac=0.3, random_state=42)
    revenue_est.to_csv(output_path / "final_tg_2024_estimation.csv", index=False)
    logger.info(f"   ✓ final_tg_2024_estimation.csv 저장")
    
    # 6. 수출액 (선택적)
    logger.info("6️⃣ 수출액 데이터 생성...")
    export_values = revenues * np.random.uniform(0, 0.5, size=n_firms)  # 매출의 0~50%
    
    export_df = pd.DataFrame({
        '업체번호': [f'biz_{i:06d}' for i in range(n_firms)],
        'export_value': export_values
    })
    
    export_df.to_csv(output_path / "export_estimation_value_final.csv", index=False)
    logger.info(f"   ✓ export_estimation_value_final.csv 저장")
    
    # 7. 자산 (선택적)
    logger.info("7️⃣ 자산 데이터 생성...")
    assets = revenues * np.random.uniform(1, 5, size=n_firms)  # 매출의 1~5배
    
    asset_df = pd.DataFrame({
        '업체번호': [f'biz_{i:06d}' for i in range(n_firms)],
        'asset': assets
    })
    
    asset_df.to_csv(output_path / "asset_final_2024_6차.csv", index=False)
    logger.info(f"   ✓ asset_final_2024_6차.csv 저장")
    
    # 8. TIS 리스크 점수 (선택적)
    logger.info("8️⃣ TIS 점수 생성...")
    tis_scores = np.random.beta(2, 5, size=n_firms)  # 0~1 사이, 낮은 값에 편중
    
    tis_df = pd.DataFrame({
        '업체번호': [f'biz_{i:06d}' for i in range(n_firms)],
        'TIS': tis_scores
    })
    
    tis_df.to_csv(output_path / "shock_after_P_v2.csv", index=False)
    logger.info(f"   ✓ shock_after_P_v2.csv 저장")
    
    logger.info("=" * 70)
    logger.info("✅ 더미 데이터 생성 완료!")
    logger.info(f"📁 출력 디렉토리: {output_path.absolute()}")
    logger.info("=" * 70)
    logger.info("\n다음 명령어로 Phase 1 실행:")
    logger.info("    python main_phase1.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="더미 데이터 생성")
    parser.add_argument('--n_firms', type=int, default=10000, help='기업 수')
    parser.add_argument('--density', type=float, default=0.01, help='H 행렬 밀도')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    generate_dummy_data(
        n_firms=args.n_firms,
        density=args.density,
        output_dir=args.output_dir
    )
