"""
Phase 3: Two-Track Hybrid Link Predict    # 데이터 경로 (현재 파일 기준)
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / "data"
    OUTPUT_DIR = DATA_DIR / "processed"
    RESULTS_DIR = SCRIPT_DIR.parent / "results" / "quick_test"(QUICK TEST)
======================================================
디버깅용 짧은 에폭 테스트

실행 방법:
    python quick_test_phase3.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import logging

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3.src.temporal_graph_builder import TemporalGraphBuilder
from phase3.src.sc_tgn import SC_TGN
from phase3.src.graphseal import GraphSEAL, HybridLinkPredictor
from phase3.src.hybrid_trainer import HybridTrainer
from phase3.src.negative_sampler import prepare_events_with_negatives

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 설정 (Config) - 빠른 테스트용
# ============================================================

class QuickConfig:
    """Phase 3 빠른 테스트 설정 (작은 모델, 적은 에폭)"""
    
    # 데이터 경로 (현재 파일 기준)
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / "data"
    OUTPUT_DIR = DATA_DIR / "processed"
    RESULTS_DIR = SCRIPT_DIR.parent / "results" / "quick_test"
    
    # 입력 파일
    NODE_EMBEDDINGS = OUTPUT_DIR / "node_embeddings_static.pt"
    TRAIN_EDGES = OUTPUT_DIR / "train_edges.npy"
    TEST_EDGES = OUTPUT_DIR / "test_edges.npy"
    TIS_SCORES = OUTPUT_DIR / "tis_score_normalized.npy"
    
    # 출력 파일
    MODEL_SAVE_PATH = RESULTS_DIR / "hybrid_model_quick.pt"
    METRICS_SAVE_PATH = RESULTS_DIR / "phase3_metrics_quick.npz"
    
    # Track A (SC-TGN) - 작은 모델
    TGN_MEMORY_DIM = 64
    TGN_TIME_DIM = 16
    TGN_MESSAGE_DIM = 64
    TGN_EMBEDDING_DIM = 32
    
    # Track B (GraphSEAL) - 작은 모델
    GRAPHSEAL_HIDDEN_DIM = 64
    GRAPHSEAL_NUM_HOPS = 1  # 1-hop만
    USE_UKGE = True
    
    # Ensemble
    ENSEMBLE_ALPHA = 0.5
    
    # Loss 설정
    LOSS_ALPHA = 0.3  # TIS 페널티 강도
    SOFT_NEGATIVE = 0.0  # Negative 엣지 soft label
    RANKING_WEIGHT = 0.1  # Ranking loss 가중치
    
    # Negative Sampling
    HISTORICAL_RATIO = 0.5  # Historical negatives 비율
    NEG_RATIO = 0.5  # Positive 1개당 Negative 개수 (빠른 테스트용)
    
    # 학습 하이퍼파라미터 - 빠른 테스트
    EPOCHS = 5  # 짧게!
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 3
    
    # 평가 설정
    RECALL_K_LIST = [10, 50, 100]
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 메인 파이프라인
# ============================================================

def main():
    """빠른 테스트 실행 함수"""
    config = QuickConfig()
    
    print("\n" + "=" * 70)
    print("🚀 Phase 3: Quick Test (5 Epochs)")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS} (빠른 테스트)")
    print(f"Track A: SC-TGN (작은 모델)")
    print(f"Track B: GraphSEAL (작은 모델)")
    print("=" * 70)
    
    try:
        # ============================================================
        # 1. 시계열 데이터 로드
        # ============================================================
        
        logger.info("\n[Step 1] 시계열 그래프 데이터 로드")
        
        builder = TemporalGraphBuilder(data_dir=str(config.DATA_DIR))
        temporal_data = builder.build_temporal_data(train_ratio=0.8)
        
        events = temporal_data['events']
        num_nodes = temporal_data['num_nodes']
        train_mask = temporal_data['train_mask']
        test_mask = temporal_data['test_mask']
        node_features = temporal_data['node_features']
        
        logger.info(f"✅ 데이터 로드 완료")
        
        # ============================================================
        # 2. Static Embeddings 로드
        # ============================================================
        
        logger.info("\n[Step 2] Static Embeddings 로드")
        
        node_embeddings = torch.load(config.NODE_EMBEDDINGS)
        train_edges = torch.from_numpy(np.load(config.TRAIN_EDGES))
        
        tis_scores = None
        if config.TIS_SCORES.exists():
            tis_scores = torch.from_numpy(np.load(config.TIS_SCORES)).float()
        
        logger.info(f"✅ 로드 완료")
        
        # ============================================================
        # 3. Train/Val 분할
        # ============================================================
        
        logger.info("\n[Step 3] Train/Val 분할")
        
        train_indices = np.where(train_mask)[0]
        np.random.shuffle(train_indices)
        
        val_size = int(len(train_indices) * 0.2)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        train_mask_final = np.zeros(len(events), dtype=bool)
        val_mask_final = np.zeros(len(events), dtype=bool)
        train_mask_final[train_indices] = True
        val_mask_final[val_indices] = True
        
        logger.info(f"   ✓ 분할 완료")
        
        # ============================================================
        # 4. 네거티브 샘플링 (Historical + Random)
        # ============================================================
        
        logger.info("\n[Step 4] 네거티브 샘플링 (Historical + Random)")
        
        train_events = prepare_events_with_negatives(
            events=events,
            mask=train_mask_final,
            num_nodes=num_nodes,
            current_edges=train_edges,
            data_dir=str(config.DATA_DIR),
            historical_ratio=config.HISTORICAL_RATIO,
            neg_ratio=config.NEG_RATIO,
            seed=42
        )
        
        val_events = prepare_events_with_negatives(
            events=events,
            mask=val_mask_final,
            num_nodes=num_nodes,
            current_edges=train_edges,
            data_dir=str(config.DATA_DIR),
            historical_ratio=config.HISTORICAL_RATIO,
            neg_ratio=config.NEG_RATIO,
            seed=43
        )
        
        test_events = prepare_events_with_negatives(
            events=events,
            mask=test_mask,
            num_nodes=num_nodes,
            current_edges=train_edges,
            data_dir=str(config.DATA_DIR),
            historical_ratio=config.HISTORICAL_RATIO,
            neg_ratio=config.NEG_RATIO,
            seed=44
        )
        
        # ============================================================
        # 5. 작은 모델 초기화
        # ============================================================
        
        logger.info("\n[Step 5] 작은 모델 초기화")
        
        # Track A: SC-TGN (작은 버전)
        tgn_model = SC_TGN(
            num_nodes=num_nodes,
            node_dim=node_features.shape[1],
            edge_dim=2,
            memory_dim=config.TGN_MEMORY_DIM,
            time_dim=config.TGN_TIME_DIM,
            message_dim=config.TGN_MESSAGE_DIM,
            embedding_dim=config.TGN_EMBEDDING_DIM
        )
        
        # Track B: GraphSEAL (작은 버전)
        graphseal_model = GraphSEAL(
            embedding_dim=node_embeddings.shape[1],
            hidden_dim=config.GRAPHSEAL_HIDDEN_DIM,
            num_hops=config.GRAPHSEAL_NUM_HOPS,
            use_ukge=config.USE_UKGE
        )
        
        # Hybrid
        hybrid_model = HybridLinkPredictor(
            tgn_model=tgn_model,
            graphseal_model=graphseal_model,
            alpha=config.ENSEMBLE_ALPHA
        )
        
        logger.info(f"✅ 모델 초기화 완료 (총 파라미터: {sum(p.numel() for p in hybrid_model.parameters()):,})")
        
        # ============================================================
        # 6. 빠른 학습
        # ============================================================
        
        logger.info("\n[Step 6] 빠른 학습 (5 epochs)")
        
        optimizer = optim.Adam(
            hybrid_model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        trainer = HybridTrainer(
            hybrid_model=hybrid_model,
            optimizer=optimizer,
            device=config.DEVICE,
            loss_alpha=config.LOSS_ALPHA,
            soft_negative=config.SOFT_NEGATIVE,
            ranking_weight=config.RANKING_WEIGHT
        )
        
        trainer.train(
            train_events=train_events,
            val_events=val_events,
            node_features=node_features,
            node_embeddings=node_embeddings,
            train_edge_index=train_edges,
            tis_scores=tis_scores,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            k_list=config.RECALL_K_LIST,
            verbose=True
        )
        
        # ============================================================
        # 7. 저장
        # ============================================================
        
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(hybrid_model.state_dict(), config.MODEL_SAVE_PATH)
        logger.info(f"\n💾 모델 저장: {config.MODEL_SAVE_PATH}")
        
        # ============================================================
        # 8. Test 평가
        # ============================================================
        
        logger.info("\n[Step 8] Test 평가")
        
        hybrid_model.tgn.reset_memory()
        
        test_metrics = trainer.evaluate(
            events=test_events,
            node_features=node_features,
            node_embeddings=node_embeddings,
            edge_index=train_edges,
            tis_scores=tis_scores,
            k_list=config.RECALL_K_LIST,
            batch_size=config.BATCH_SIZE * 2
        )
        
        # ============================================================
        # 9. 결과 저장
        # ============================================================
        
        np.savez(
            config.METRICS_SAVE_PATH,
            test_metrics=test_metrics,
            train_losses=trainer.train_losses,
            val_losses=trainer.val_losses,
            val_recalls=trainer.val_recalls
        )
        
        logger.info(f"💾 메트릭 저장: {config.METRICS_SAVE_PATH}")
        
        # ============================================================
        # 10. 완료
        # ============================================================
        
        print("\n" + "=" * 70)
        print("✅ Quick Test 완료!")
        print("=" * 70)
        print(f"\n📊 Test 성능 (Recall@K):")
        for k in config.RECALL_K_LIST:
            print(f"   - Recall@{k}: {test_metrics[f'recall@{k}']:.4f}")
        print("=" * 70)
        print("\n다음 단계:")
        print("  python main_phase3_hybrid.py  # 전체 학습 (100 epochs)")
        print("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
        logger.info("\n💡 TIP: Phase 1, 2를 먼저 실행하세요")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
