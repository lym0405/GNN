"""
Phase 2: Static Graph Embedding with Curriculum Learning
=========================================================
커리큘럼 학습 기반 GraphSAGE 임베딩 생성

실행 방법:
    python main_phase2_fixed.py
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

from phase2.src.graph_builder import StaticGraphBuilder
from phase2.src.sampler import CurriculumNegativeSampler
from phase2.src.GraphSAGE import GraphSAGE
from phase2.src.loss import RiskAwareBCELoss
from phase2.src.trainer import CurriculumTrainer

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
    """Phase 2 설정"""
    
    # 데이터 경로 (현재 파일 기준)
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / "data"
    OUTPUT_DIR = DATA_DIR / "processed"
    
    # 출력 파일
    NODE_EMBEDDINGS = OUTPUT_DIR / "node_embeddings_static.pt"
    TRAIN_EDGES = OUTPUT_DIR / "train_edges.npy"
    TEST_EDGES = OUTPUT_DIR / "test_edges.npy"
    
    # 피처 설정
    USE_SIMPLE_FEATURES = True  # True: 73차원, False: 197차원
    
    # 모델 하이퍼파라미터
    HIDDEN_DIM = 64
    OUTPUT_DIM = 32
    DROPOUT = 0.3
    
    # 학습 하이퍼파라미터
    EPOCHS = 60
    BATCH_SIZE = 4096  # 배치 크기 증가 (학습 속도 향상)
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # 커리큘럼 설정
    EASY_EPOCHS = 20     # Epoch 1-20: Random only
    MEDIUM_EPOCHS = 30   # Epoch 21-50: 20% Historical
    HARD_EPOCHS = 5      # Epoch 51-55: 40% Historical
    FINAL_EPOCHS = 5     # Epoch 56-60: 30% Historical
    
    # 데이터 분할
    TRAIN_RATIO = 0.8    # Train/Test split ratio
    RANDOM_SEED = 42
    
    # TIS Loss
    TIS_ALPHA = 0.3      # TIS 페널티 강도
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 메인 파이프라인
# ============================================================

def split_train_test_edges(
    edge_index: torch.Tensor,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> tuple:
    """
    엣지를 Train/Test로 분할 (Data Leakage 방지)
    
    Returns
    -------
    train_edges : torch.Tensor
    test_edges : torch.Tensor
    """
    logger.info(f"📊 Train/Test 엣지 분할 ({train_ratio*100:.0f}/{(1-train_ratio)*100:.0f})")
    
    np.random.seed(random_seed)
    num_edges = edge_index.shape[1]
    indices = np.random.permutation(num_edges)
    
    split_idx = int(num_edges * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_edges = edge_index[:, train_indices]
    test_edges = edge_index[:, test_indices]
    
    logger.info(f"   ✓ Train 엣지: {train_edges.shape[1]:,}")
    logger.info(f"   ✓ Test 엣지: {test_edges.shape[1]:,}")
    
    return train_edges, test_edges


def main():
    """메인 실행 함수"""
    config = Config()
    
    print("\n" + "=" * 70)
    print("🚀 Phase 2: Static Graph Embedding (Curriculum Learning)")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Feature Dim: {'73 (Simple)' if config.USE_SIMPLE_FEATURES else '197 (Full)'}")
    print(f"Curriculum: Easy(20)→Medium(30)→Hard(5)→Final(5) = {config.EPOCHS} epochs")
    print("=" * 70)
    
    try:
        # ============================================================
        # 1. 그래프 데이터 로드
        # ============================================================
        
        logger.info("\n[Step 1] 그래프 데이터 로드")
        builder = StaticGraphBuilder(data_dir=str(config.DATA_DIR), use_cache=True)
        
        X, edge_index, edge_attr, firm_ids = builder.build_static_data(
            use_simple_features=config.USE_SIMPLE_FEATURES
        )
        
        N = X.shape[0]
        D = X.shape[1]
        E = edge_index.shape[1]
        
        logger.info(f"✅ 그래프 데이터 로드 완료")
        logger.info(f"   - 노드 수: {N:,}")
        logger.info(f"   - 피처 차원: {D}")
        logger.info(f"   - 엣지 수: {E:,}")
        
        # 피처 저장
        builder.save_features(X)
        
        # ============================================================
        # 2. Train/Test 분할
        # ============================================================
        
        logger.info("\n[Step 2] Train/Test 분할")
        train_edges, test_edges = split_train_test_edges(
            edge_index,
            train_ratio=config.TRAIN_RATIO,
            random_seed=config.RANDOM_SEED
        )
        
        # 저장
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.TRAIN_EDGES, train_edges.numpy())
        np.save(config.TEST_EDGES, test_edges.numpy())
        logger.info(f"💾 엣지 저장 완료")
        
        # Validation 분할 (Train의 20%)
        train_val_split = int(train_edges.shape[1] * 0.8)
        train_pos_edges = train_edges[:, :train_val_split]
        val_pos_edges = train_edges[:, train_val_split:]
        
        logger.info(f"   ✓ Train Pos: {train_pos_edges.shape[1]:,}")
        logger.info(f"   ✓ Val Pos: {val_pos_edges.shape[1]:,}")
        
        # ============================================================
        # 3. TIS 점수 로드
        # ============================================================
        
        logger.info("\n[Step 3] TIS 점수 로드")
        tis_path = config.OUTPUT_DIR / "tis_score_normalized.npy"
        if tis_path.exists():
            tis_scores = torch.from_numpy(np.load(tis_path)).float().squeeze()
            logger.info(f"✅ TIS 로드: {tis_scores.shape}")
        else:
            logger.warning(f"⚠️  TIS 파일 없음, TIS 없이 학습")
            tis_scores = None
        
        # ============================================================
        # 4. 모델 초기화
        # ============================================================
        
        logger.info("\n[Step 4] 모델 초기화")
        
        model = GraphSAGE(
            in_dim=D,
            hidden_dim=config.HIDDEN_DIM,
            out_dim=config.OUTPUT_DIM,
            dropout=config.DROPOUT
        )
        
        logger.info(f"✅ GraphSAGE 모델")
        logger.info(f"   - 입력 차원: {D}")
        logger.info(f"   - 은닉 차원: {config.HIDDEN_DIM}")
        logger.info(f"   - 출력 차원: {config.OUTPUT_DIM}")
        logger.info(f"   - 파라미터: {sum(p.numel() for p in model.parameters()):,}")
        
        # ============================================================
        # 5. 네거티브 샘플러 초기화
        # ============================================================
        
        logger.info("\n[Step 5] 네거티브 샘플러 초기화")
        
        sampler = CurriculumNegativeSampler(
            num_nodes=N,
            edge_index=train_edges,  # Train 엣지만 사용!
            data_dir=str(config.DATA_DIR)
        )
        
        # ============================================================
        # 6. 학습
        # ============================================================
        
        logger.info("\n[Step 6] 학습 시작")
        
        loss_fn = RiskAwareBCELoss(alpha=config.TIS_ALPHA)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        trainer = CurriculumTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=config.DEVICE
        )
        
        trainer.train(
            x=X,
            edge_index=train_edges,  # Message Passing은 Train 엣지만!
            train_pos_edges=train_pos_edges,
            val_pos_edges=val_pos_edges,
            sampler=sampler,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            tis_scores=tis_scores
        )
        
        # ============================================================
        # 7. 임베딩 저장
        # ============================================================
        
        logger.info("\n[Step 7] 임베딩 저장")
        
        model.eval()
        with torch.no_grad():
            X_device = X.to(config.DEVICE)
            train_edges_device = train_edges.to(config.DEVICE)
            embeddings = model(X_device, train_edges_device)
            embeddings = embeddings.cpu()
        
        torch.save(embeddings, config.NODE_EMBEDDINGS)
        logger.info(f"💾 임베딩 저장: {config.NODE_EMBEDDINGS}")
        logger.info(f"   - Shape: {embeddings.shape}")
        
        # ============================================================
        # 8. 완료
        # ============================================================
        
        print("\n" + "=" * 70)
        print("✅ Phase 2 완료!")
        print("=" * 70)
        print(f"📁 출력 파일:")
        print(f"   - {config.NODE_EMBEDDINGS}")
        print(f"   - {config.TRAIN_EDGES}")
        print(f"   - {config.TEST_EDGES}")
        print("=" * 70)
        print("\n다음 단계:")
        print("  python main_phase3_train.py  # Phase 3 학습")
        print("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
        logger.info("\n💡 TIP: Phase 1을 먼저 실행하세요:")
        logger.info("   python main_phase1.py")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
