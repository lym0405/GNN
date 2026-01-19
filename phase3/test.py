"""
Phase 3: Link Prediction Quick Test (DEBUG VERSION)
====================================================
링크 예측 빠른 테스트 (짧은 에폭)

실행 방법:
    python test_phase3.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.link_predictor import MLPLinkPredictor
from src.phase3_trainer import LinkPredictionTrainer, prepare_dataloader
from src.metrics import evaluate_link_prediction, convert_torch_to_numpy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 설정 (Quick Test Config)
# ============================================================

class QuickTestConfig:
    """Phase 3 빠른 테스트 설정 (짧은 에폭)"""
    
    # 데이터 경로
    DATA_DIR = Path("data")
    OUTPUT_DIR = DATA_DIR / "processed"
    RESULTS_DIR = Path("results")
    
    # 입력 파일
    NODE_EMBEDDINGS = OUTPUT_DIR / "node_embeddings_static.pt"
    TRAIN_EDGES = OUTPUT_DIR / "train_edges.npy"
    TEST_EDGES = OUTPUT_DIR / "test_edges.npy"
    
    # 출력 파일
    MODEL_SAVE_PATH = RESULTS_DIR / "link_predictor_quicktest.pt"
    METRICS_SAVE_PATH = RESULTS_DIR / "phase3_quicktest_metrics.npz"
    
    # 모델 하이퍼파라미터 (간단한 구조)
    HIDDEN_DIMS = [64, 32]  # 더 작은 네트워크
    DROPOUT = 0.3
    AGGREGATION = 'concat'
    
    # 학습 하이퍼파라미터 (짧은 학습)
    EPOCHS = 10  # ⭐ 빠른 테스트용
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 5
    
    # 네거티브 샘플링
    NEG_SAMPLING_RATIO = 1.0
    
    # 평가 설정
    EVAL_THRESHOLD = 0.5
    TOPK_LIST = [10, 50, 100]
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 유틸리티 함수 (main_phase3.py와 동일)
# ============================================================

def sample_negative_edges(
    num_nodes: int,
    positive_edges: torch.Tensor,
    num_negatives: int,
    seed: int = 42
) -> torch.Tensor:
    """네거티브 엣지 샘플링"""
    logger.info(f"🎲 네거티브 엣지 샘플링: {num_negatives:,}개")
    
    np.random.seed(seed)
    pos_set = set(map(tuple, positive_edges.t().numpy()))
    
    negative_edges = []
    attempts = 0
    max_attempts = num_negatives * 10
    
    while len(negative_edges) < num_negatives and attempts < max_attempts:
        u = np.random.randint(0, num_nodes, size=num_negatives * 2)
        v = np.random.randint(0, num_nodes, size=num_negatives * 2)
        
        for i in range(len(u)):
            if u[i] != v[i] and (u[i], v[i]) not in pos_set:
                negative_edges.append([u[i], v[i]])
                if len(negative_edges) >= num_negatives:
                    break
        
        attempts += 1
    
    negative_edges = torch.tensor(negative_edges[:num_negatives]).t()
    logger.info(f"   ✓ 네거티브 샘플링 완료: {negative_edges.shape[1]:,}개")
    
    return negative_edges


def prepare_train_val_data(
    train_edges: torch.Tensor,
    num_nodes: int,
    neg_ratio: float = 1.0,
    val_ratio: float = 0.2,
    seed: int = 42
):
    """Train/Val 데이터 준비"""
    logger.info("\n[데이터 준비] Train/Val 분할")
    
    num_edges = train_edges.shape[1]
    num_val = int(num_edges * val_ratio)
    num_train = num_edges - num_val
    
    np.random.seed(seed)
    indices = np.random.permutation(num_edges)
    
    train_pos_edges = train_edges[:, indices[:num_train]]
    val_pos_edges = train_edges[:, indices[num_train:]]
    
    logger.info(f"   ✓ Train Positive: {train_pos_edges.shape[1]:,}")
    logger.info(f"   ✓ Val Positive: {val_pos_edges.shape[1]:,}")
    
    num_train_neg = int(train_pos_edges.shape[1] * neg_ratio)
    num_val_neg = int(val_pos_edges.shape[1] * neg_ratio)
    
    train_neg_edges = sample_negative_edges(
        num_nodes, train_edges, num_train_neg, seed=seed
    )
    val_neg_edges = sample_negative_edges(
        num_nodes, train_edges, num_val_neg, seed=seed + 1
    )
    
    train_edge_index = torch.cat([train_pos_edges, train_neg_edges], dim=1)
    train_labels = torch.cat([
        torch.ones(train_pos_edges.shape[1]),
        torch.zeros(train_neg_edges.shape[1])
    ])
    
    val_edge_index = torch.cat([val_pos_edges, val_neg_edges], dim=1)
    val_labels = torch.cat([
        torch.ones(val_pos_edges.shape[1]),
        torch.zeros(val_neg_edges.shape[1])
    ])
    
    logger.info(f"   ✓ Train Total: {train_edge_index.shape[1]:,}")
    logger.info(f"   ✓ Val Total: {val_edge_index.shape[1]:,}")
    
    return train_edge_index, train_labels, val_edge_index, val_labels


def prepare_test_data(
    test_edges: torch.Tensor,
    num_nodes: int,
    train_edges: torch.Tensor,
    neg_ratio: float = 1.0,
    seed: int = 42
):
    """Test 데이터 준비"""
    logger.info("\n[데이터 준비] Test 데이터")
    
    num_test_pos = test_edges.shape[1]
    num_test_neg = int(num_test_pos * neg_ratio)
    
    logger.info(f"   ✓ Test Positive: {num_test_pos:,}")
    
    all_pos_edges = torch.cat([train_edges, test_edges], dim=1)
    test_neg_edges = sample_negative_edges(
        num_nodes, all_pos_edges, num_test_neg, seed=seed + 2
    )
    
    test_edge_index = torch.cat([test_edges, test_neg_edges], dim=1)
    test_labels = torch.cat([
        torch.ones(num_test_pos),
        torch.zeros(num_test_neg)
    ])
    
    logger.info(f"   ✓ Test Total: {test_edge_index.shape[1]:,}")
    
    return test_edge_index, test_labels


# ============================================================
# 메인 파이프라인 (Quick Test)
# ============================================================

def main():
    """빠른 테스트 메인 함수"""
    config = QuickTestConfig()
    
    print("\n" + "=" * 70)
    print("🧪 Phase 3: Link Prediction (QUICK TEST - 10 Epochs)")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS} ⚡ (빠른 테스트)")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Hidden Dims: {config.HIDDEN_DIMS}")
    print("=" * 70)
    
    try:
        # ============================================================
        # 1. 데이터 로드
        # ============================================================
        
        logger.info("\n[Step 1] 데이터 로드")
        
        embeddings = torch.load(config.NODE_EMBEDDINGS)
        num_nodes = embeddings.shape[0]
        emb_dim = embeddings.shape[1]
        
        logger.info(f"✅ 임베딩 로드: {embeddings.shape}")
        
        train_edges = torch.from_numpy(np.load(config.TRAIN_EDGES))
        test_edges = torch.from_numpy(np.load(config.TEST_EDGES))
        
        logger.info(f"✅ Train 엣지: {train_edges.shape[1]:,}")
        logger.info(f"✅ Test 엣지: {test_edges.shape[1]:,}")
        
        # ============================================================
        # 2. Train/Val/Test 데이터 준비
        # ============================================================
        
        train_edge_index, train_labels, val_edge_index, val_labels = \
            prepare_train_val_data(
                train_edges, num_nodes, config.NEG_SAMPLING_RATIO,
                val_ratio=0.2, seed=42
            )
        
        test_edge_index, test_labels = prepare_test_data(
            test_edges, num_nodes, train_edges,
            config.NEG_SAMPLING_RATIO, seed=42
        )
        
        train_loader = prepare_dataloader(
            train_edge_index, train_labels, config.BATCH_SIZE, shuffle=True
        )
        val_loader = prepare_dataloader(
            val_edge_index, val_labels, config.BATCH_SIZE, shuffle=False
        )
        test_loader = prepare_dataloader(
            test_edge_index, test_labels, config.BATCH_SIZE, shuffle=False
        )
        
        # ============================================================
        # 3. 모델 초기화
        # ============================================================
        
        logger.info("\n[Step 3] 모델 초기화")
        
        model = MLPLinkPredictor(
            input_dim=emb_dim,
            hidden_dims=config.HIDDEN_DIMS,
            dropout=config.DROPOUT,
            aggregation=config.AGGREGATION
        )
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        logger.info(f"✅ MLPLinkPredictor")
        logger.info(f"   - 입력 차원: {emb_dim}")
        logger.info(f"   - 은닉 차원: {config.HIDDEN_DIMS}")
        logger.info(f"   - 파라미터: {sum(p.numel() for p in model.parameters()):,}")
        
        # ============================================================
        # 4. 학습 (10 Epochs만)
        # ============================================================
        
        logger.info("\n[Step 4] 학습 시작 (빠른 테스트)")
        
        trainer = LinkPredictionTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=config.DEVICE
        )
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            embeddings=embeddings,
            epochs=config.EPOCHS,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            verbose=True
        )
        
        # ============================================================
        # 5. Test 평가
        # ============================================================
        
        logger.info("\n[Step 5] Test 평가")
        
        test_scores = trainer.predict(
            test_edge_index, embeddings, batch_size=config.BATCH_SIZE
        )
        
        test_labels_np = test_labels.numpy()
        
        test_metrics = evaluate_link_prediction(
            y_true=test_labels_np,
            y_score=test_scores,
            threshold=config.EVAL_THRESHOLD,
            k_list=config.TOPK_LIST,
            verbose=True
        )
        
        # ============================================================
        # 6. 결과 저장
        # ============================================================
        
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        logger.info(f"\n💾 모델 저장: {config.MODEL_SAVE_PATH}")
        
        np.savez(
            config.METRICS_SAVE_PATH,
            test_scores=test_scores,
            test_labels=test_labels_np,
            test_metrics=test_metrics,
            train_losses=trainer.train_losses,
            val_losses=trainer.val_losses,
            val_accs=trainer.val_accs
        )
        
        logger.info(f"💾 메트릭 저장: {config.METRICS_SAVE_PATH}")
        
        # ============================================================
        # 7. 완료
        # ============================================================
        
        print("\n" + "=" * 70)
        print("✅ Phase 3 빠른 테스트 완료! 🧪")
        print("=" * 70)
        print(f"📁 출력 파일:")
        print(f"   - {config.MODEL_SAVE_PATH}")
        print(f"   - {config.METRICS_SAVE_PATH}")
        print("=" * 70)
        print(f"\n📊 Test 성능 (10 Epochs):")
        print(f"   - Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"   - Precision: {test_metrics['precision']:.4f}")
        print(f"   - Recall:    {test_metrics['recall']:.4f}")
        print(f"   - F1 Score:  {test_metrics['f1']:.4f}")
        print(f"   - AUC-ROC:   {test_metrics.get('auc_roc', 0):.4f}")
        print("=" * 70)
        print("\n다음 명령어로 전체 학습:")
        print("  python main_phase3.py  # 100 epochs 전체 학습")
        print("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
        logger.info("\n💡 TIP: Phase 2를 먼저 실행하세요:")
        logger.info("   python main_phase2.py")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
