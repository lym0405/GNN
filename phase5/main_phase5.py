"""
Phase 5: Historical Back-testing (MAIN)
========================================
2019년 일본 수출규제 시뮬레이션 및 검증

실행 방법:
    python main_phase5.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase5.src.ksic_matcher import KSICMatcher, JapanExportRestriction2019
from phase5.src.shock_injector import ShockInjector, create_shock_scenario
from phase5.src.evaluator import Phase5Evaluator, ResilienceEvaluator, compare_networks

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
    """Phase 5 설정"""
    
    # 데이터 경로
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / "data"
    RAW_DIR = DATA_DIR / "raw"
    OUTPUT_DIR = DATA_DIR / "processed"
    RESULTS_DIR = SCRIPT_DIR.parent / "results" / "phase5"
    
    # 입력 파일
    NETWORK_2018 = RAW_DIR / "posco_network_capital_consumergoods_removed_2018.csv"
    NETWORK_2019 = RAW_DIR / "posco_network_capital_consumergoods_removed_2019.csv"
    NETWORK_2020 = RAW_DIR / "posco_network_capital_consumergoods_removed_2020.csv"
    
    FIRM_INFO = RAW_DIR / "firm_info.csv"  # KSIC 코드 포함
    FIRM_TO_IDX = RAW_DIR / "firm_to_idx_model2.csv"
    
    # Phase 3 모델 (재배선 예측용)
    PHASE3_MODEL = SCRIPT_DIR.parent / "results" / "hybrid_model_best.pt"
    NODE_EMBEDDINGS = OUTPUT_DIR / "node_embeddings_static.pt"
    
    # 출력 파일
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_FILE = RESULTS_DIR / "predictions_2019_shock.npz"
    METRICS_FILE = RESULTS_DIR / "evaluation_metrics.npz"
    REPORT_FILE = RESULTS_DIR / "phase5_report.txt"
    
    # 시나리오 설정
    SHOCK_TYPE = 'edge_deletion'  # 'edge_deletion', 'node_disruption', 'supply_cut'
    SHOCK_INTENSITY = 1.0  # 0.0~1.0
    
    # 평가 설정
    K_LIST = [10, 50, 100, 500, 1000]
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 메인 파이프라인
# ============================================================

def main():
    """메인 실행 함수"""
    config = Config()
    
    print("\n" + "=" * 70)
    print("🔬 Phase 5: Historical Back-testing")
    print("=" * 70)
    print("Scenario: 2019년 일본 수출규제 (반도체 핵심 소재)")
    print("=" * 70)
    
    try:
        # ============================================================
        # Step 1: 데이터 로드
        # ============================================================
        
        logger.info("\n[Step 1] 데이터 로드")
        
        # 네트워크 데이터
        if not config.NETWORK_2018.exists():
            logger.error(f"❌ 2018년 네트워크 파일 없음: {config.NETWORK_2018}")
            logger.info("💡 TIP: data/raw/ 디렉토리에 2018, 2019, 2020년 네트워크 파일 필요")
            return
        
        network_2018 = pd.read_csv(config.NETWORK_2018)
        logger.info(f"   ✓ 2018년 네트워크: {len(network_2018):,}개 엣지")
        
        if config.NETWORK_2020.exists():
            network_2020 = pd.read_csv(config.NETWORK_2020)
            logger.info(f"   ✓ 2020년 네트워크: {len(network_2020):,}개 엣지")
        else:
            logger.warning(f"   ⚠️  2020년 네트워크 파일 없음")
            network_2020 = None
        
        # ============================================================
        # Step 2: KSIC 매칭 (충격 대상 기업 선정)
        # ============================================================
        
        logger.info("\n[Step 2] KSIC 매칭 (충격 대상 기업 선정)")
        
        # KSIC Matcher 초기화
        if not config.FIRM_INFO.exists():
            logger.warning(f"   ⚠️  기업 정보 파일 없음: {config.FIRM_INFO}")
            logger.info("   💡 KSIC 매칭을 건너뛰고 알려진 기업만 사용합니다")
            matcher = None
        else:
            matcher = KSICMatcher(
                firm_info_path=str(config.FIRM_INFO),
                firm_to_idx_path=str(config.FIRM_TO_IDX)
            )
        
        # 시나리오 설정 로드
        scenario = JapanExportRestriction2019.get_scenario_config()
        
        logger.info(f"   ✓ 시나리오: {scenario['name']}")
        logger.info(f"   ✓ 충격 날짜: {scenario['shock_date']}")
        logger.info(f"   ✓ 대상 소재: {', '.join(scenario['target_materials'])}")
        
        # 공급자 인덱스 추출
        if matcher:
            supplier_indices = matcher.get_firm_indices_by_ksic(
                ksic_codes=scenario['supplier_ksic'],
                exact_match=False
            )
        else:
            # 알려진 기업만 사용
            supplier_mapping = {}
            if config.FIRM_TO_IDX.exists():
                firm_to_idx_df = pd.read_csv(config.FIRM_TO_IDX)
                firm_to_idx = dict(zip(
                    firm_to_idx_df['사업자등록번호'].astype(str),
                    firm_to_idx_df['idx']
                ))
                for firm_id in scenario['known_suppliers']:
                    if firm_id in firm_to_idx:
                        supplier_mapping[firm_id] = firm_to_idx[firm_id]
            
            supplier_indices = list(supplier_mapping.values())
        
        logger.info(f"   ✓ 공급자: {len(supplier_indices):,}개 기업")
        
        # 수요자 인덱스 추출
        if matcher:
            buyer_indices = matcher.get_firm_indices_by_ksic(
                ksic_codes=scenario['buyer_ksic'],
                exact_match=False
            )
        else:
            buyer_mapping = {}
            if config.FIRM_TO_IDX.exists():
                for firm_id in scenario['known_buyers']:
                    if firm_id in firm_to_idx:
                        buyer_mapping[firm_id] = firm_to_idx[firm_id]
            
            buyer_indices = list(buyer_mapping.values())
        
        logger.info(f"   ✓ 수요자: {len(buyer_indices):,}개 기업")
        
        # ============================================================
        # Step 3: 충격 시나리오 주입
        # ============================================================
        
        logger.info("\n[Step 3] 충격 시나리오 주입")
        
        shocked_network, shocked_features, shock_info = create_shock_scenario(
            network_df=network_2018,
            supplier_indices=supplier_indices,
            buyer_indices=buyer_indices,
            shock_type=config.SHOCK_TYPE,
            shock_intensity=config.SHOCK_INTENSITY
        )
        
        logger.info(f"   ✓ 충격 타입: {config.SHOCK_TYPE}")
        logger.info(f"   ✓ 충격 강도: {config.SHOCK_INTENSITY:.1%}")
        
        # ============================================================
        # Step 4: 재배선 예측 (Phase 3 모델 사용)
        # ============================================================
        
        logger.info("\n[Step 4] 재배선 예측")
        
        # Phase 3 모델 로드 (구현 필요)
        # 여기서는 placeholder로 랜덤 예측
        logger.warning("   ⚠️  Phase 3 모델 통합은 추후 구현 예정")
        logger.info("   💡 현재는 랜덤 예측을 사용합니다")
        
        # 랜덤 예측 생성 (placeholder)
        num_predictions = 1000
        predicted_edges = []
        
        np.random.seed(42)
        for _ in range(num_predictions):
            src = np.random.choice(supplier_indices) if supplier_indices else np.random.randint(0, 1000)
            dst = np.random.choice(buyer_indices) if buyer_indices else np.random.randint(0, 1000)
            score = np.random.rand()
            predicted_edges.append((src, dst, score))
        
        logger.info(f"   ✓ 예측 엣지: {len(predicted_edges):,}개")
        
        # 예측 저장
        np.savez(
            config.PREDICTIONS_FILE,
            predicted_edges=np.array(predicted_edges, dtype=object),
            shock_info=shock_info
        )
        logger.info(f"   💾 예측 저장: {config.PREDICTIONS_FILE}")
        
        # ============================================================
        # Step 5: 실제 데이터와 비교 (2020년)
        # ============================================================
        
        if network_2020 is not None:
            logger.info("\n[Step 5] 실제 데이터와 비교 (2020년)")
            
            # 신규/삭제 엣지 추출
            new_edges, removed_edges = compare_networks(network_2018, network_2020)
            
            logger.info(f"   ✓ 2018 → 2020 변화:")
            logger.info(f"      - 신규 엣지: {len(new_edges):,}개")
            logger.info(f"      - 삭제 엣지: {len(removed_edges):,}개")
            
            # ============================================================
            # Step 6: 성능 평가
            # ============================================================
            
            logger.info("\n[Step 6] 성능 평가")
            
            evaluator = Phase5Evaluator(
                predicted_edges=predicted_edges,
                actual_new_edges=new_edges,
                actual_removed_edges=removed_edges
            )
            
            metrics = evaluator.compute_all_metrics(k_list=config.K_LIST)
            
            # 메트릭 저장
            np.savez(
                config.METRICS_FILE,
                **metrics
            )
            logger.info(f"   💾 메트릭 저장: {config.METRICS_FILE}")
            
            # ============================================================
            # Step 7: 결과 보고서 생성
            # ============================================================
            
            logger.info("\n[Step 7] 결과 보고서 생성")
            
            report = generate_report(
                scenario=scenario,
                shock_info=shock_info,
                metrics=metrics,
                config=config
            )
            
            with open(config.REPORT_FILE, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"   💾 보고서 저장: {config.REPORT_FILE}")
            
            # 콘솔 출력
            print("\n" + report)
        
        else:
            logger.warning("\n⚠️  2020년 데이터 없음, 평가 건너뜀")
        
        # ============================================================
        # 완료
        # ============================================================
        
        print("\n" + "=" * 70)
        print("✅ Phase 5 완료!")
        print("=" * 70)
        print(f"📁 출력 파일:")
        print(f"   - 예측: {config.PREDICTIONS_FILE}")
        if network_2020 is not None:
            print(f"   - 메트릭: {config.METRICS_FILE}")
            print(f"   - 보고서: {config.REPORT_FILE}")
        print("=" * 70)
    
    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
        logger.info("\n💡 필요한 파일:")
        logger.info("   1. data/raw/posco_network_2018.csv")
        logger.info("   2. data/raw/posco_network_2020.csv")
        logger.info("   3. data/raw/firm_to_idx_model2.csv")
        logger.info("   4. data/raw/firm_info.csv (optional, KSIC 매칭용)")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_report(
    scenario: Dict,
    shock_info: Dict,
    metrics: Dict,
    config: Config
) -> str:
    """결과 보고서 생성"""
    
    report = []
    report.append("=" * 70)
    report.append("Phase 5: Historical Back-testing 결과 보고서")
    report.append("=" * 70)
    report.append("")
    
    # 시나리오 정보
    report.append("📋 시나리오 정보")
    report.append("-" * 70)
    report.append(f"이름: {scenario['name']}")
    report.append(f"설명: {scenario['description']}")
    report.append(f"충격 날짜: {scenario['shock_date']}")
    report.append(f"대상 소재: {', '.join(scenario['target_materials'])}")
    report.append("")
    
    # 충격 정보
    report.append("🔨 충격 시나리오")
    report.append("-" * 70)
    report.append(f"타입: {shock_info.get('type', 'N/A')}")
    
    if 'edge_deletion' in shock_info:
        edge_info = shock_info['edge_deletion']
        report.append(f"삭제된 엣지: {edge_info.get('deleted_edges_count', 0):,}개")
        report.append(f"남은 엣지: {edge_info.get('remaining_edges_count', 0):,}개")
    
    report.append("")
    
    # 평가 결과
    report.append("📊 평가 결과")
    report.append("-" * 70)
    
    # Hit Rate
    report.append("\n[1] Hit Rate@K (Top-K에서 실제 신규 엣지 포착 비율)")
    for k in config.K_LIST:
        key = f'hit_rate@{k}'
        if key in metrics:
            report.append(f"   Hit Rate@{k:4d}: {metrics[key]:.4f}")
    
    # Recall
    report.append("\n[2] Recall@K (실제 신규 엣지 중 예측 비율)")
    for k in config.K_LIST:
        key = f'recall@{k}'
        if key in metrics:
            report.append(f"   Recall@{k:4d}: {metrics[key]:.4f}")
    
    # Precision
    report.append("\n[3] Precision@K (예측 중 실제 신규 엣지 비율)")
    for k in config.K_LIST:
        key = f'precision@{k}'
        if key in metrics:
            report.append(f"   Precision@{k:4d}: {metrics[key]:.4f}")
    
    report.append("")
    
    # 통계
    report.append("📈 통계")
    report.append("-" * 70)
    report.append(f"총 예측 엣지: {metrics.get('total_predictions', 0):,}개")
    report.append(f"실제 신규 엣지: {metrics.get('total_actual_new', 0):,}개")
    report.append(f"실제 삭제 엣지: {metrics.get('total_actual_removed', 0):,}개")
    report.append("")
    
    report.append("=" * 70)
    
    return "\n".join(report)


if __name__ == "__main__":
    main()
