"""
Phase 1: Production Recipe Estimation
======================================
기업별 생산함수(33차원 레시피) 추정

실행 순서:
1. 데이터 로드 (IO 테이블, H 행렬, 기업정보, 매출)
2. BMatrixGenerator로 B 행렬 생성
3. ZeroShotInventoryModule로 레시피 추정
4. 검증 및 저장

실행 방법:
    python main_phase1.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import pickle
import logging

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase1.src.b_matrix_generator import BMatrixGenerator
from phase1.src.inventory_module import ZeroShotInventoryModule
from phase1.src.check_recipe import RecipeValidator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 설정 (Config)
# ============================================================

class Config:
    """Phase 1 설정"""
    
    # 데이터 경로 (현재 파일 기준)
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / "data" / "raw"
    OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "processed"
    
    # 입력 파일
    IO_TABLE = DATA_DIR / "A_33.csv"
    H_MATRIX = DATA_DIR / "H_csr_model2.npz"
    FIRM_TO_IDX = DATA_DIR / "firm_to_idx_model2.csv"
    FIRM_INFO = DATA_DIR / "vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv"
    REVENUE = DATA_DIR / "tg_2024_filtered.csv"
    REVENUE_EST = DATA_DIR / "final_tg_2024_estimation.csv"
    
    # 출력 파일
    B_MATRIX_OUTPUT = OUTPUT_DIR / "B_matrix.npy"
    RECIPE_OUTPUT = OUTPUT_DIR / "disentangled_recipes.pkl"
    RECIPE_CSV = OUTPUT_DIR / "recipes_dataframe.csv"
    VALIDATION_REPORT = OUTPUT_DIR / "recipe_validation_report.csv"
    
    # 하이퍼파라미터
    ESTIMATION_METHOD = 'weighted'  # 'weighted', 'simple', 'bayesian'
    USE_REVENUE_WEIGHTING = True
    BATCH_SIZE = 10000


# ============================================================
# 메인 파이프라인
# ============================================================

def load_data(config: Config):
    """데이터 로드"""
    logger.info("=" * 70)
    logger.info("📂 Phase 1: 데이터 로드 시작")
    logger.info("=" * 70)
    
    # 1. IO 테이블
    logger.info("1️⃣ IO 테이블 로드...")
    A_df = pd.read_csv(config.IO_TABLE, index_col=0)
    A_matrix = A_df.values.astype(np.float32)
    logger.info(f"   ✓ Shape: {A_matrix.shape}")
    logger.info(f"   ✓ 산업명: {list(A_df.columns[:5])}...")
    
    # 2. H 행렬 (거래 네트워크)
    logger.info("2️⃣ H 행렬 로드...")
    H_matrix = load_npz(config.H_MATRIX)
    logger.info(f"   ✓ Shape: {H_matrix.shape}")
    logger.info(f"   ✓ 밀도: {H_matrix.nnz / (H_matrix.shape[0] ** 2) * 100:.4f}%")
    logger.info(f"   ✓ 총 거래 금액: {H_matrix.sum() / 1e12:.2f} 조원")
    
    # 3. 기업 인덱스 매핑
    logger.info("3️⃣ 기업 인덱스 매핑 로드...")
    firm_to_idx = pd.read_csv(config.FIRM_TO_IDX)
    firm_ids = firm_to_idx['사업자등록번호'].astype(str).tolist()
    logger.info(f"   ✓ 기업 수: {len(firm_ids)}")
    
    # 4. 기업 정보 (산업코드)
    logger.info("4️⃣ 기업 정보 로드...")
    firm_info = pd.read_csv(config.FIRM_INFO)
    logger.info(f"   ✓ 기업 정보 레코드: {len(firm_info)}")
    logger.info(f"   ✓ 컬럼: {list(firm_info.columns[:10])}...")
    
    # 5. 매출 데이터
    logger.info("5️⃣ 매출 데이터 로드...")
    
    # 실제 매출
    if config.REVENUE.exists():
        revenue_df = pd.read_csv(config.REVENUE)
        logger.info(f"   ✓ 실제 매출 데이터: {len(revenue_df)} 기업")
    else:
        revenue_df = None
        logger.warning(f"   ⚠️ 실제 매출 파일 없음: {config.REVENUE}")
    
    # 추정 매출
    if config.REVENUE_EST.exists():
        revenue_est_df = pd.read_csv(config.REVENUE_EST)
        logger.info(f"   ✓ 추정 매출 데이터: {len(revenue_est_df)} 기업")
    else:
        revenue_est_df = None
        logger.warning(f"   ⚠️ 추정 매출 파일 없음: {config.REVENUE_EST}")
    
    # 매출 데이터 병합
    if revenue_df is not None and revenue_est_df is not None:
        revenue_final = pd.concat([revenue_df, revenue_est_df]).drop_duplicates(subset=['업체번호'], keep='first')
    elif revenue_df is not None:
        revenue_final = revenue_df
    elif revenue_est_df is not None:
        revenue_final = revenue_est_df
    else:
        logger.error("❌ 매출 데이터가 없습니다!")
        revenue_final = None
    
    logger.info("=" * 70)
    
    return {
        'A_matrix': A_matrix,
        'H_matrix': H_matrix,
        'firm_ids': firm_ids,
        'firm_info': firm_info,
        'revenue': revenue_final,
    }


def build_sector_mapping(firm_info: pd.DataFrame, firm_ids: list):
    """
    기업 → 산업 매핑 생성
    
    structure 문서 기준:
    - IO상품_단일_대분류_코드: Phase 1 레시피 추정용 (33개 대분류)
    
    Returns
    -------
    biz_sector_map : dict
        {사업자등록번호: 산업인덱스(0~32)}
    """
    logger.info("🔧 기업-산업 매핑 생성 중...")
    
    biz_sector_map = {}
    
    # 사업자등록번호를 키로 매핑
    firm_info['사업자등록번호'] = firm_info['사업자등록번호'].astype(str)
    
    for _, row in firm_info.iterrows():
        biz_no = row['사업자등록번호']
        
        # IO 상품 코드 추출 (컬럼명 우선순위)
        sector_code = None
        
        # 1순위: IO상품_단일_대분류_코드 (실제 데이터, structure 문서 기준)
        if 'IO상품_단일_대분류_코드' in row and pd.notna(row['IO상품_단일_대분류_코드']):
            sector_code = str(row['IO상품_단일_대분류_코드']).strip()
        else:
            # 2순위: IO상품 관련 컬럼 (부분 매칭)
            for col in firm_info.columns:
                if 'IO상품' in col and '단일' in col and '대분류' in col and '코드' in col:
                    if pd.notna(row[col]):
                        sector_code = str(row[col]).strip()
                        break
            
            # 3순위: 더미 데이터용 컬럼명
            if sector_code is None:
                for col in ['산업코드', 'sector_code', 'industry_code', 'io_sector']:
                    if col in row and pd.notna(row[col]):
                        sector_code = str(row[col])
                        break
        
        if sector_code:
            try:
                # IO 상품 코드를 인덱스로 변환
                # IO 코드는 1~33 범위이므로 0-based index로 변환
                sector_idx = int(sector_code)
                
                # 1-based index라면 0-based로 변환
                if 1 <= sector_idx <= 33:
                    sector_idx = sector_idx - 1
                
                # 0-based index가 유효한지 확인
                if 0 <= sector_idx < 33:
                    biz_sector_map[biz_no] = sector_idx
            except (ValueError, TypeError):
                # 변환 실패 시 무시
                pass
    
    logger.info(f"   ✓ 매핑 완료: {len(biz_sector_map)} 기업")
    
    # 매핑되지 않은 기업 처리 (기본값: -1)
    for firm_id in firm_ids:
        if firm_id not in biz_sector_map:
            biz_sector_map[firm_id] = -1  # Unknown
    
    logger.info(f"   ✓ 전체 기업: {len(firm_ids)}")
    logger.info(f"   ✓ 매핑 실패: {sum(1 for v in biz_sector_map.values() if v == -1)}")
    
    return biz_sector_map


def build_revenue_share(revenue: pd.DataFrame, biz_sector_map: dict):
    """
    산업별 매출 점유율 계산
    
    structure 문서 기준:
    - final_tg_2024_estimation.csv: tg_2024_final (최종 매출액)
    
    Returns
    -------
    biz_share_map : dict
        {사업자등록번호: 산업 내 점유율(0~1)}
    """
    logger.info("💰 산업별 매출 점유율 계산 중...")
    
    # 업체번호 정규화
    revenue['업체번호'] = revenue['업체번호'].astype(str)
    
    # 매출 컬럼 찾기 (우선순위 기반)
    revenue_col = None
    
    # 1순위: tg_2024_final (structure 문서 기준)
    if 'tg_2024_final' in revenue.columns:
        revenue_col = 'tg_2024_final'
    else:
        # 2순위: 기타 매출 관련 컬럼
        for col in ['tg_2024', 'revenue', 'sales', 'total_sales', '매출액']:
            if col in revenue.columns:
                revenue_col = col
                logger.info(f"   📋 매출 컬럼: '{revenue_col}' 사용 (tg_2024_final 없음)")
                break
    
    if not revenue_col:
        logger.warning("   ⚠️ 매출 컬럼을 찾을 수 없습니다. 점유율 계산 생략.")
        logger.warning(f"   사용 가능한 컬럼: {list(revenue.columns[:10])}")
        return {}
    
    # 산업별 매출 집계
    sector_revenues = {}
    for _, row in revenue.iterrows():
        firm_id = row['업체번호']
        rev = float(row[revenue_col]) if pd.notna(row[revenue_col]) else 0
        
        if firm_id in biz_sector_map and rev > 0:
            sector = biz_sector_map[firm_id]
            if sector >= 0:
                sector_revenues.setdefault(sector, []).append((firm_id, rev))
    
    # 점유율 계산
    biz_share_map = {}
    for sector, firms in sector_revenues.items():
        total_revenue = sum(r for _, r in firms)
        for firm_id, rev in firms:
            biz_share_map[firm_id] = rev / total_revenue
    
    logger.info(f"   ✓ 점유율 계산 완료: {len(biz_share_map)} 기업")
    
    return biz_share_map


def generate_B_matrix(config: Config, data: dict):
    """B 행렬 생성"""
    logger.info("=" * 70)
    logger.info("🔨 B 행렬 생성")
    logger.info("=" * 70)
    
    # 산업 매핑
    biz_sector_map = build_sector_mapping(data['firm_info'], data['firm_ids'])
    
    # 매출 점유율
    if config.USE_REVENUE_WEIGHTING and data['revenue'] is not None:
        biz_share_map = build_revenue_share(data['revenue'], biz_sector_map)
    else:
        biz_share_map = {}
    
    # BMatrixGenerator 초기화
    generator = BMatrixGenerator(
        A_matrix=data['A_matrix'],
        biz_sector_map=biz_sector_map,
        biz_share_map=biz_share_map
    )
    
    # B 행렬 생성
    B_matrix = generator.generate_B_matrix(data['firm_ids'])
    
    # 저장
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generator.save_B_matrix(B_matrix, str(config.B_MATRIX_OUTPUT))
    
    logger.info("=" * 70)
    
    return B_matrix


def estimate_recipes(config: Config, data: dict, B_matrix: np.ndarray):
    """레시피 추정"""
    logger.info("=" * 70)
    logger.info("🧪 레시피 추정")
    logger.info("=" * 70)
    
    # ZeroShotInventoryModule 초기화
    module = ZeroShotInventoryModule(
        H_matrix=data['H_matrix'],
        B_matrix=B_matrix,
        firm_ids=data['firm_ids']
    )
    
    # 레시피 추정
    recipes = module.estimate_recipes(method=config.ESTIMATION_METHOD)
    
    # 저장 (pickle)
    module.save_recipes(recipes, str(config.RECIPE_OUTPUT))
    
    # 저장 (CSV)
    recipe_df = module.export_to_dataframe(recipes)
    recipe_df.to_csv(config.RECIPE_CSV, index=False)
    logger.info(f"💾 레시피 DataFrame 저장: {config.RECIPE_CSV}")
    
    logger.info("=" * 70)
    
    return recipes


def validate_recipes(config: Config):
    """레시피 검증"""
    logger.info("=" * 70)
    logger.info("✅ 레시피 검증")
    logger.info("=" * 70)
    
    # 레시피 로드
    with open(config.RECIPE_OUTPUT, 'rb') as f:
        recipe_dict = pickle.load(f)
    
    # 검증 수행
    validator = RecipeValidator(recipe_dict)
    validator.run_all_checks()
    
    # 리포트 저장
    validator.export_report(str(config.VALIDATION_REPORT))
    
    logger.info("=" * 70)


def main():
    """메인 실행 함수"""
    config = Config()
    
    print("\n" + "=" * 70)
    print("🚀 Phase 1: Production Recipe Estimation")
    print("=" * 70)
    
    try:
        # 1. 데이터 로드
        data = load_data(config)
        
        # 2. B 행렬 생성
        B_matrix = generate_B_matrix(config, data)
        
        # 3. 레시피 추정
        recipes = estimate_recipes(config, data, B_matrix)
        
        # 4. 검증
        validate_recipes(config)
        
        print("\n" + "=" * 70)
        print("✅ Phase 1 완료!")
        print("=" * 70)
        print(f"📁 출력 파일:")
        print(f"   - {config.RECIPE_OUTPUT}")
        print(f"   - {config.RECIPE_CSV}")
        print(f"   - {config.VALIDATION_REPORT}")
        print("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
        logger.info("\n💡 TIP: data/raw/ 폴더에 필요한 데이터를 배치해주세요:")
        logger.info("   - A_33.csv")
        logger.info("   - H_csr_model2.npz")
        logger.info("   - firm_to_idx_model2.csv")
        logger.info("   - vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv")
        logger.info("   - tg_2024_filtered.csv")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
