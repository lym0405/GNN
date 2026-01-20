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
from phase1.src.product_matcher import ProductMatcher, create_io_product_dict
from phase1.src.attention_disentangler import create_disentangled_recipes

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
    REVENUE = DATA_DIR / "final_tg_2024_estimation.csv"
    
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
    
    # 추정 매출 (final_tg_2024_estimation.csv만 존재)
    if config.REVENUE_EST.exists():
        revenue_final = pd.read_csv(config.REVENUE_EST)
        logger.info(f"   ✓ 추정 매출 데이터: {len(revenue_final)} 기업")
    else:
        # 실제 매출 파일 확인 (폴백)
        if config.REVENUE.exists():
            revenue_final = pd.read_csv(config.REVENUE)
            logger.info(f"   ✓ 실제 매출 데이터: {len(revenue_final)} 기업")
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


def generate_B_matrix(config: Config, data: dict):
    """B 행렬 생성"""
    logger.info("=" * 70)
    logger.info("🔨 B 행렬 생성")
    logger.info("=" * 70)
    
    # BMatrixGenerator는 파일 경로를 받아야 함
    # (내부에서 직접 데이터를 로드하는 구조)
    generator = BMatrixGenerator(
        io_path=str(config.IO_TABLE),
        h_path=str(config.H_MATRIX),
        firm_info_path=str(config.FIRM_INFO),
        sales_path=str(config.REVENUE),  # 추정 매출 파일
        alpha=0.5
    )
    
    # get_vector 메서드를 사용하여 각 기업의 레시피 생성
    logger.info("   - 기업별 레시피 생성 중...")
    B_matrix = []
    none_count = 0
    for firm_id in data['firm_ids']:
        recipe = generator.get_vector(firm_id)
        if recipe is not None:
            B_matrix.append(recipe)
        else:
            # 매핑 실패 시 제로 벡터
            B_matrix.append(np.zeros(33))
            none_count += 1
    
    B_matrix = np.array(B_matrix)
    
    logger.info(f"   ✓ B 행렬 생성 완료: {B_matrix.shape}")
    logger.info(f"   ✓ 매핑 성공: {len(data['firm_ids']) - none_count}/{len(data['firm_ids'])} 기업")
    
    # 저장
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(config.B_MATRIX_OUTPUT), B_matrix)
    logger.info(f"   ✓ 저장: {config.B_MATRIX_OUTPUT}")
    
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


def estimate_recipes_with_attention(config: Config, data: dict, B_matrix: np.ndarray):
    """
    [NEW] Attention 기반 레시피 추정
    
    단계:
    1. ProductMatcher로 기업별 주요상품 매칭
    2. AttentionDisentangler로 Query-Key Attention 수행
    3. 다중 상품 레시피 분리
    """
    logger.info("=" * 70)
    logger.info("🧪 Attention 기반 레시피 추정 (Zero-Shot Inventory Module)")
    logger.info("=" * 70)
    
    # 1. IO 상품 딕셔너리 생성
    logger.info("1️⃣ IO 상품 딕셔너리 생성...")
    io_dict = create_io_product_dict(str(config.IO_TABLE))
    logger.info(f"   ✓ IO 상품 {len(io_dict)}개")
    
    # 2. ProductMatcher로 기업별 상품 매칭
    logger.info("2️⃣ 기업별 주요상품 매칭...")
    matcher = ProductMatcher(io_dict)
    
    firm_products = matcher.batch_match(
        df_firms=data['firm_info'],
        col_product_text='주요상품목록',
        col_multi_code='IO상품_다중_대분류_코드',
        use_multi_code=True,
        top_k=3
    )
    
    # 3. Attention으로 레시피 분리
    logger.info("3️⃣ Attention 기반 레시피 분리...")
    recipes = create_disentangled_recipes(
        H_matrix=data['H_matrix'],
        B_matrix=B_matrix,
        firm_products=firm_products,
        firm_ids=data['firm_ids'],
        method='attention',
        temperature=0.8,  # Temperature (작을수록 sharp)
        alpha=0.7  # Attention vs Prior 가중치
    )
    
    logger.info(f"   ✓ 레시피 생성 완료: {recipes.shape}")
    
    # 4. 저장
    logger.info("4️⃣ 레시피 저장...")
    
    # Pickle 저장
    with open(str(config.RECIPE_OUTPUT), 'wb') as f:
        pickle.dump({
            'recipes': recipes,
            'firm_ids': data['firm_ids'],
            'firm_products': firm_products,
            'method': 'attention',
            'config': {
                'temperature': 0.8,
                'alpha': 0.7
            }
        }, f)
    logger.info(f"   ✓ Pickle 저장: {config.RECIPE_OUTPUT}")
    
    # CSV 저장
    df_recipes = pd.DataFrame(
        recipes,
        index=data['firm_ids'],
        columns=[f"IO_{i+1:02d}" for i in range(33)]
    )
    df_recipes.to_csv(str(config.RECIPE_CSV))
    logger.info(f"   ✓ CSV 저장: {config.RECIPE_CSV}")
    
    logger.info("=" * 70)
    
    return recipes, firm_products


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
