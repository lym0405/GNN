"""
Phase 2 Simple Test
===================
Phase 2를 빠르게 테스트하는 Python 스크립트

실행 방법:
    python test_phase2.py
"""

import sys
import os
from pathlib import Path

def test_phase2():
    """Phase 2 간단 테스트"""
    
    print("=" * 70)
    print("🧪 Phase 2 Simple Test")
    print("=" * 70)
    
    # 1. 더미 데이터 생성
    print("\n[1/3] 더미 데이터 생성...")
    
    try:
        from generate_phase2_dummy_data import generate_phase2_test_data
        generate_phase2_test_data(n_firms=500, density=0.03)
        print("✅ 더미 데이터 생성 완료")
    except Exception as e:
        print(f"❌ 더미 데이터 생성 실패: {e}")
        return False
    
    # 2. Phase 2 실행 (축소 버전)
    print("\n[2/3] Phase 2 실행 (10 epochs, 빠른 테스트)...")
    
    try:
        # Config 임시 수정
        sys.path.insert(0, str(Path(__file__).parent))
        
        from main_phase2_fixed import Config, main
        
        # 에폭 수 줄이기
        original_epochs = Config.EPOCHS
        Config.EPOCHS = 10
        Config.EASY_EPOCHS = 7
        Config.MEDIUM_EPOCHS = 2
        Config.HARD_EPOCHS = 1
        Config.FINAL_EPOCHS = 0
        Config.BATCH_SIZE = 256  # 배치 크기도 줄임
        
        print(f"   ⚙️  Config: {Config.EPOCHS} epochs (원래: {original_epochs})")
        
        # 실행
        main()
        
        print("✅ Phase 2 실행 완료")
    except Exception as e:
        print(f"❌ Phase 2 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 결과 검증
    print("\n[3/3] 결과 검증...")
    
    success = True
    
    # 임베딩 확인
    emb_path = Path("data/processed/node_embeddings_static.pt")
    if emb_path.exists():
        import torch
        emb = torch.load(emb_path)
        print(f"✅ 임베딩:")
        print(f"   - Shape: {emb.shape}")
        print(f"   - Norm: {emb.norm(dim=1).mean():.4f}")
        print(f"   - 범위: [{emb.min():.3f}, {emb.max():.3f}]")
    else:
        print("❌ 임베딩 파일 없음")
        success = False
    
    # 엣지 확인
    train_edge_path = Path("data/processed/train_edges.npy")
    test_edge_path = Path("data/processed/test_edges.npy")
    
    if train_edge_path.exists() and test_edge_path.exists():
        import numpy as np
        train_edges = np.load(train_edge_path)
        test_edges = np.load(test_edge_path)
        
        print(f"✅ 엣지:")
        print(f"   - Train: {train_edges.shape}")
        print(f"   - Test: {test_edges.shape}")
        print(f"   - 비율: {train_edges.shape[1]/(train_edges.shape[1]+test_edges.shape[1])*100:.1f}%")
        
        # Overlap 체크
        train_set = set(map(tuple, train_edges.T))
        test_set = set(map(tuple, test_edges.T))
        overlap = train_set & test_set
        
        if len(overlap) == 0:
            print(f"   - Overlap: 0 (✅ Data Leakage 없음)")
        else:
            print(f"   - Overlap: {len(overlap)} (⚠️ Data Leakage 발생!)")
            success = False
    else:
        print("❌ 엣지 파일 없음")
        success = False
    
    # 피처 확인
    feature_path = Path("data/processed/X_feature_matrix.npy")
    if feature_path.exists():
        import numpy as np
        X = np.load(feature_path)
        print(f"✅ 피처:")
        print(f"   - Shape: {X.shape}")
        print(f"   - NaN: {np.isnan(X).sum()}")
        print(f"   - 범위: [{X.min():.3f}, {X.max():.3f}]")
    else:
        print("❌ 피처 파일 없음")
        success = False
    
    # 최종 결과
    print("\n" + "=" * 70)
    if success:
        print("✅ 모든 테스트 통과!")
        print("=" * 70)
        print("\n다음 단계:")
        print("  python main_phase2_fixed.py  # 전체 실행 (60 epochs)")
    else:
        print("❌ 일부 테스트 실패")
        print("=" * 70)
    
    return success


if __name__ == "__main__":
    try:
        success = test_phase2()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
