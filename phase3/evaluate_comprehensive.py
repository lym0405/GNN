"""
Phase 3: Comprehensive Evaluation with Benchmarks
==================================================
GNN 모델 vs 휴리스틱 벤치마크 (PA, RA, JC) 비교
MRR, RMSE, Recall@K 종합 평가

실행 방법:
    python evaluate_phase3_comprehensive.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import logging

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3.src.temporal_graph_builder import TemporalGraphBuilder
from phase3.src.sc_tgn import SC_TGN
from phase3.src.graphseal import GraphSEAL, HybridLinkPredictor
from phase3.src.benchmarks import evaluate_benchmarks
from phase3.src.metrics import evaluate_link_prediction_comprehensive
from phase3.src.robustness_test import run_robustness_test, visualize_robustness_results

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """종합 평가 실행"""
    
    print("\n" + "=" * 70)
    print("📊 Phase 3: Comprehensive Evaluation")
    print("=" * 70)
    print("1. GNN 모델 평가 (MRR, RMSE, Recall@K)")
    print("2. 벤치마크 비교 (PA, RA, JC)")
    print("3. 견고성 테스트 (옵션)")
    print("=" * 70)
    
    # 경로 설정
    DATA_DIR = Path("data")
    OUTPUT_DIR = DATA_DIR / "processed"
    RESULTS_DIR = Path("results")
    
    MODEL_PATH = RESULTS_DIR / "hybrid_model_best.pt"
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # ============================================================
        # 1. 데이터 로드
        # ============================================================
        
        logger.info("\n[Step 1] 데이터 로드")
        
        # 시계열 데이터
        builder = TemporalGraphBuilder(data_dir=str(DATA_DIR))
        temporal_data = builder.build_temporal_data(train_ratio=0.8)
        
        events = temporal_data['events']
        num_nodes = temporal_data['num_nodes']
        test_mask = temporal_data['test_mask']
        node_features = temporal_data['node_features']
        
        # Static embeddings
        node_embeddings = torch.load(OUTPUT_DIR / "node_embeddings_static.pt")
        train_edges = torch.from_numpy(np.load(OUTPUT_DIR / "train_edges.npy"))
        test_edges = torch.from_numpy(np.load(OUTPUT_DIR / "test_edges.npy"))
        
        # TIS 점수
        tis_path = OUTPUT_DIR / "tis_score_normalized.npy"
        if tis_path.exists():
            tis_scores = torch.from_numpy(np.load(tis_path)).float()
        else:
            tis_scores = None
        
        logger.info(f"✅ 데이터 로드 완료")
        
        # ============================================================
        # 2. 모델 로드
        # ============================================================
        
        logger.info("\n[Step 2] 모델 로드")
        
        if not MODEL_PATH.exists():
            logger.error(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
            logger.info("💡 먼저 학습을 실행하세요: python main_phase3_hybrid.py")
            sys.exit(1)
        
        # 모델 초기화 (학습 시와 동일한 설정)
        tgn_model = SC_TGN(
            num_nodes=num_nodes,
            node_dim=node_features.shape[1],
            edge_dim=2,
            memory_dim=128,
            time_dim=32,
            message_dim=128,
            embedding_dim=64
        )
        
        graphseal_model = GraphSEAL(
            embedding_dim=node_embeddings.shape[1],
            hidden_dim=128,
            num_hops=2,
            use_ukge=True
        )
        
        hybrid_model = HybridLinkPredictor(
            tgn_model=tgn_model,
            graphseal_model=graphseal_model,
            alpha=0.5
        )
        
        # 가중치 로드
        hybrid_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        hybrid_model.to(DEVICE)
        hybrid_model.eval()
        
        logger.info(f"✅ 모델 로드 완료: {MODEL_PATH}")
        
        # ============================================================
        # 3. Test 데이터 준비
        # ============================================================
        
        logger.info("\n[Step 3] Test 데이터 준비")
        
        from phase3.src.negative_sampler import prepare_events_with_negatives
        
        test_events = prepare_events_with_negatives(
            events=events,
            mask=test_mask,
            num_nodes=num_nodes,
            current_edges=train_edges,
            data_dir=str(DATA_DIR),
            historical_ratio=0.5,
            neg_ratio=1.0,
            seed=44
        )
        
        # ============================================================
        # 4. GNN 모델 평가 (MRR, RMSE, Recall@K)
        # ============================================================
        
        logger.info("\n[Step 4] GNN 모델 종합 평가")
        
        # Forward pass
        all_scores = []
        all_labels = []
        all_confidences = []
        all_tis = []
        
        with torch.no_grad():
            hybrid_model.tgn.reset_memory()
            
            batch_size = 2048
            for i in range(0, len(test_events), batch_size):
                batch_events = test_events[i:i+batch_size]
                
                timestamps = torch.tensor([e[0] for e in batch_events], dtype=torch.long)
                src_nodes = torch.tensor([e[1] for e in batch_events], dtype=torch.long)
                dst_nodes = torch.tensor([e[2] for e in batch_events], dtype=torch.long)
                labels = torch.tensor([e[4] for e in batch_events], dtype=torch.float32)
                
                timestamps = timestamps.to(DEVICE)
                src_nodes = src_nodes.to(DEVICE)
                dst_nodes = dst_nodes.to(DEVICE)
                
                src_features = node_features[src_nodes].to(DEVICE)
                dst_features = node_features[dst_nodes].to(DEVICE)
                
                # Forward
                logits, outputs = hybrid_model(
                    src_nodes=src_nodes,
                    dst_nodes=dst_nodes,
                    src_features=src_features,
                    dst_features=dst_features,
                    node_embeddings=node_embeddings.to(DEVICE),
                    edge_index=train_edges.to(DEVICE),
                    timestamps=timestamps,
                    tis_scores=None
                )
                
                scores = torch.sigmoid(logits)
                confidences = outputs.get('confidence', torch.ones_like(scores))
                
                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels.numpy())
                all_confidences.append(confidences.cpu().numpy())
                
                # TIS (dst 노드 기준)
                if tis_scores is not None:
                    batch_tis = tis_scores[dst_nodes.cpu()].numpy()
                    all_tis.append(batch_tis)
        
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        all_confidences = np.concatenate(all_confidences)
        all_tis = np.concatenate(all_tis) if len(all_tis) > 0 else None
        
        # 종합 평가
        gnn_metrics = evaluate_link_prediction_comprehensive(
            y_true=all_labels,
            y_score=all_scores,
            confidence_scores=all_confidences,
            tis_scores=all_tis,
            k_list=[10, 50, 100, 500, 1000],
            threshold=0.5,
            alpha=0.3
        )
        
        # ============================================================
        # 5. 벤치마크 휴리스틱 평가 (PA, RA, JC)
        # ============================================================
        
        logger.info("\n[Step 5] 벤치마크 휴리스틱 평가")
        
        # Test 엣지 추출
        test_src = torch.tensor([e[1] for e in test_events], dtype=torch.long)
        test_dst = torch.tensor([e[2] for e in test_events], dtype=torch.long)
        test_labels_tensor = torch.tensor([e[4] for e in test_events], dtype=torch.float32)
        
        test_pos_edges = torch.stack([
            test_src[test_labels_tensor == 1],
            test_dst[test_labels_tensor == 1]
        ])
        
        test_neg_edges = torch.stack([
            test_src[test_labels_tensor == 0],
            test_dst[test_labels_tensor == 0]
        ])
        
        benchmark_results = evaluate_benchmarks(
            edge_index=train_edges,
            num_nodes=num_nodes,
            test_pos_edges=test_pos_edges,
            test_neg_edges=test_neg_edges,
            k_list=[10, 50, 100]
        )
        
        # ============================================================
        # 6. 결과 비교 테이블
        # ============================================================
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 GNN vs Benchmarks 성능 비교")
        logger.info("=" * 70)
        
        # 테이블 출력
        print("\n" + "=" * 80)
        print(f"{'Model':<15} {'Recall@10':<12} {'Recall@50':<12} {'Recall@100':<12} {'MRR':<12}")
        print("=" * 80)
        
        # GNN 모델
        print(
            f"{'GNN (Ours)':<15} "
            f"{gnn_metrics['recall@10']:<12.4f} "
            f"{gnn_metrics['recall@50']:<12.4f} "
            f"{gnn_metrics['recall@100']:<12.4f} "
            f"{gnn_metrics['MRR']:<12.4f}"
        )
        
        # 벤치마크들
        for method, metrics in benchmark_results.items():
            print(
                f"{method:<15} "
                f"{metrics.get('recall@10', 0.0):<12.4f} "
                f"{metrics.get('recall@50', 0.0):<12.4f} "
                f"{metrics.get('recall@100', 0.0):<12.4f} "
                f"{metrics.get('MRR', 0.0):<12.4f}"
            )
        
        print("=" * 80)
        
        # RMSE 결과
        print("\n📊 RMSE (Risk-aware Prediction Error):")
        print("-" * 40)
        print(f"  - RMSE (Overall): {gnn_metrics.get('rmse', 0.0):.4f}")
        print(f"  - RMSE (Positive): {gnn_metrics.get('rmse_pos', 0.0):.4f}")
        print(f"  - RMSE (Negative): {gnn_metrics.get('rmse_neg', 0.0):.4f}")
        if 'rmse_tis_aware' in gnn_metrics:
            print(f"  - RMSE (TIS-aware): {gnn_metrics['rmse_tis_aware']:.4f}")
        if 'rmse_weighted' in gnn_metrics:
            print(f"  - RMSE (Confidence-weighted): {gnn_metrics['rmse_weighted']:.4f}")
        print("-" * 40)
        
        # ============================================================
        # 7. 견고성 테스트 (옵션)
        # ============================================================
        
        print("\n💡 견고성 테스트를 실행하시겠습니까? (y/n)")
        choice = input("선택: ").strip().lower()
        
        if choice == 'y':
            logger.info("\n[Step 7] 견고성 테스트 실행")
            
            robustness_results = run_robustness_test(
                model=hybrid_model,
                events=events,
                node_features=node_features,
                node_embeddings=node_embeddings,
                edge_index=train_edges,
                tis_scores=tis_scores if tis_scores is not None else torch.zeros(num_nodes),
                test_mask=test_mask,
                num_nodes=num_nodes,
                device=DEVICE,
                neg_ratios=[1.0, 2.0, 3.0, 4.0]
            )
            
            # 시각화
            visualize_robustness_results(
                robustness_results,
                save_path=str(RESULTS_DIR / "robustness_test.png")
            )
        
        # ============================================================
        # 8. 결과 저장
        # ============================================================
        
        logger.info("\n[Step 8] 결과 저장")
        
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            RESULTS_DIR / "comprehensive_evaluation.npz",
            gnn_metrics=gnn_metrics,
            benchmark_results=benchmark_results,
            test_scores=all_scores,
            test_labels=all_labels,
            test_confidences=all_confidences
        )
        
        logger.info(f"💾 결과 저장: {RESULTS_DIR / 'comprehensive_evaluation.npz'}")
        
        # ============================================================
        # 9. 완료
        # ============================================================
        
        print("\n" + "=" * 70)
        print("✅ 종합 평가 완료!")
        print("=" * 70)
        print(f"📁 결과 파일:")
        print(f"   - {RESULTS_DIR / 'comprehensive_evaluation.npz'}")
        if choice == 'y':
            print(f"   - {RESULTS_DIR / 'robustness_test.png'}")
        print("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
