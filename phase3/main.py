"""
Phase 3: Two-Track Hybrid Link Prediction (MAIN - Full Training)
=================================================================
Track A (SC-TGN) + Track B (GraphSEAL) + Ensemble

실행 방법:
    python main_phase3.py
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
# 설정 (Config)
# ============================================================

class Config:
    """Phase 3 전체 학습 설정"""
    
    # 데이터 경로 (현재 파일 기준)
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / "data"
    OUTPUT_DIR = DATA_DIR / "processed"
    RESULTS_DIR = SCRIPT_DIR.parent / "results"
    
    # 입력 파일
    NODE_EMBEDDINGS = OUTPUT_DIR / "node_embeddings_static.pt"
    TRAIN_EDGES = OUTPUT_DIR / "train_edges.npy"
    TEST_EDGES = OUTPUT_DIR / "test_edges.npy"
    TIS_SCORES = OUTPUT_DIR / "tis_score_normalized.npy"
    
    # 출력 파일
    MODEL_SAVE_PATH = RESULTS_DIR / "hybrid_model_best.pt"
    METRICS_SAVE_PATH = RESULTS_DIR / "phase3_metrics.npz"
    
    # Track A (SC-TGN) 하이퍼파라미터
    TGN_MEMORY_DIM = 128
    TGN_TIME_DIM = 32
    TGN_MESSAGE_DIM = 128
    TGN_EMBEDDING_DIM = 64
    
    # Track B (GraphSEAL) 하이퍼파라미터
    GRAPHSEAL_HIDDEN_DIM = 128
    GRAPHSEAL_NUM_HOPS = 2
    USE_UKGE = True
    
    # Ensemble
    ENSEMBLE_ALPHA = 0.5  # Track A 가중치 (초기값, 학습됨)
    
    # Loss 설정
    LOSS_ALPHA = 0.3  # TIS 페널티 강도
    SOFT_NEGATIVE = 0.0  # Negative 엣지 soft label (0.0 or 0.05)
    RANKING_WEIGHT = 0.1  # Ranking loss 가중치
    
    # Negative Sampling
    HISTORICAL_RATIO = 0.5  # Historical negatives 비율 (0.0~1.0)
    NEG_RATIO = 1.0  # Positive 1개당 Negative 개수
    
    # 학습 하이퍼파라미터
    EPOCHS = 100
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 15
    
    # 평가 설정
    RECALL_K_LIST = [10, 50, 100, 500, 1000]
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 메인 파이프라인
# ============================================================

def main():
    """메인 실행 함수"""
    config = Config()
    
    print("\n" + "=" * 70)
    print("🚀 Phase 3: Two-Track Hybrid Link Prediction (FULL TRAINING)")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Track A: SC-TGN (Temporal)")
    print(f"Track B: GraphSEAL (Structural + UKGE)")
    print("=" * 70)
    
    try:
        # ============================================================
        # 1. 시계열 그래프 데이터 로드
        # ============================================================
        
        logger.info("\n[Step 1] 시계열 그래프 데이터 로드")
        
        builder = TemporalGraphBuilder(data_dir=str(config.DATA_DIR), use_cache=True)
        temporal_data = builder.build_temporal_data(train_ratio=0.8)
        
        events = temporal_data['events']
        num_nodes = temporal_data['num_nodes']
        train_mask = temporal_data['train_mask']
        test_mask = temporal_data['test_mask']
        node_features = temporal_data['node_features']
        
        logger.info(f"✅ 시계열 데이터 로드 완료")
        logger.info(f"   - 노드 수: {num_nodes}")
        logger.info(f"   - 총 이벤트: {len(events):,}")
        logger.info(f"   - 노드 피처 차원: {node_features.shape[1]}")
        
        # ============================================================
        # 2. Phase 2 출력 (Static Embeddings) 로드
        # ============================================================
        
        logger.info("\n[Step 2] Static Embeddings 로드 (Phase 2 출력)")
        
        node_embeddings = torch.load(config.NODE_EMBEDDINGS)
        logger.info(f"✅ 임베딩 로드: {node_embeddings.shape}")
        
        # Train 엣지 인덱스 (GraphSEAL용)
        train_edges = torch.from_numpy(np.load(config.TRAIN_EDGES))
        logger.info(f"✅ Train 엣지: {train_edges.shape[1]:,}")
        
        # TIS 점수 로드
        if config.TIS_SCORES.exists():
            tis_scores = torch.from_numpy(np.load(config.TIS_SCORES)).float()
            logger.info(f"✅ TIS 점수 로드: {tis_scores.shape}")
        else:
            tis_scores = None
            logger.warning("⚠️  TIS 점수 없음")
        
        # ============================================================
        # 3. Train/Val 분할
        # ============================================================
        
        logger.info("\n[Step 3] Train/Val 분할")
        
        # Train에서 20%를 Val로
        train_indices = np.where(train_mask)[0]
        np.random.shuffle(train_indices)
        
        val_size = int(len(train_indices) * 0.2)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        # 새로운 마스크
        train_mask_final = np.zeros(len(events), dtype=bool)
        val_mask_final = np.zeros(len(events), dtype=bool)
        train_mask_final[train_indices] = True
        val_mask_final[val_indices] = True
        
        logger.info(f"   ✓ Train: {train_mask_final.sum():,}")
        logger.info(f"   ✓ Val: {val_mask_final.sum():,}")
        logger.info(f"   ✓ Test: {test_mask.sum():,}")
        
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
        # 5. 모델 초기화
        # ============================================================
        
        logger.info("\n[Step 5] 모델 초기화")
        
        # Track A: SC-TGN
        tgn_model = SC_TGN(
            num_nodes=num_nodes,
            node_dim=node_features.shape[1],
            edge_dim=2,  # edge feature 차원
            memory_dim=config.TGN_MEMORY_DIM,
            time_dim=config.TGN_TIME_DIM,
            message_dim=config.TGN_MESSAGE_DIM,
            embedding_dim=config.TGN_EMBEDDING_DIM
        )
        
        logger.info(f"✅ Track A (SC-TGN)")
        logger.info(f"   - Memory Dim: {config.TGN_MEMORY_DIM}")
        logger.info(f"   - Embedding Dim: {config.TGN_EMBEDDING_DIM}")
        logger.info(f"   - 파라미터: {sum(p.numel() for p in tgn_model.parameters()):,}")
        
        # Track B: GraphSEAL
        graphseal_model = GraphSEAL(
            embedding_dim=node_embeddings.shape[1],
            hidden_dim=config.GRAPHSEAL_HIDDEN_DIM,
            num_hops=config.GRAPHSEAL_NUM_HOPS,
            use_ukge=config.USE_UKGE
        )
        
        logger.info(f"✅ Track B (GraphSEAL)")
        logger.info(f"   - Hidden Dim: {config.GRAPHSEAL_HIDDEN_DIM}")
        logger.info(f"   - Num Hops: {config.GRAPHSEAL_NUM_HOPS}")
        logger.info(f"   - UKGE: {config.USE_UKGE}")
        logger.info(f"   - 파라미터: {sum(p.numel() for p in graphseal_model.parameters()):,}")
        
        # Hybrid Model
        hybrid_model = HybridLinkPredictor(
            tgn_model=tgn_model,
            graphseal_model=graphseal_model,
            alpha=config.ENSEMBLE_ALPHA
        )
        
        logger.info(f"✅ Hybrid Model (Ensemble)")
        logger.info(f"   - 초기 Alpha: {config.ENSEMBLE_ALPHA}")
        logger.info(f"   - 총 파라미터: {sum(p.numel() for p in hybrid_model.parameters()):,}")
        
        # ============================================================
        # 6. 학습
        # ============================================================
        
        logger.info("\n[Step 6] 학습 시작")
        
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
        # 7. 모델 저장
        # ============================================================
        
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(hybrid_model.state_dict(), config.MODEL_SAVE_PATH)
        logger.info(f"\n💾 모델 저장: {config.MODEL_SAVE_PATH}")
        
        # ============================================================
        # 8. Test 평가
        # ============================================================
        
        logger.info("\n[Step 8] Test 평가")
        
        # TGN 메모리 초기화
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
        
        logger.info("\n[Step 9] 결과 저장")
        
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
        print("✅ Phase 3 완료!")
        print("=" * 70)
        print(f"📁 출력 파일:")
        print(f"   - {config.MODEL_SAVE_PATH}")
        print(f"   - {config.METRICS_SAVE_PATH}")
        print("=" * 70)
        print(f"\n📊 최종 Test 성능 (Recall@K):")
        for k in config.RECALL_K_LIST:
            print(f"   - Recall@{k}: {test_metrics[f'recall@{k}']:.4f}")
        print("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
        logger.info("\n💡 TIP: Phase 1과 Phase 2를 먼저 실행하세요:")
        logger.info("   python main_phase1.py")
        logger.info("   python main_phase2.py")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
