#!/bin/bash
# Phase 2 빠른 테스트 스크립트

echo "=========================================="
echo "Phase 2: Quick Test (Curriculum Learning)"
echo "=========================================="

# 1. 패키지 설치
echo ""
echo "📦 1단계: 필수 패키지 설치..."
echo "   - Phase 1 패키지..."
pip install -q numpy pandas scipy matplotlib

echo "   - Phase 2 패키지 (PyTorch)..."
# CPU 버전 설치 (빠른 테스트용)
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -q torch-geometric
pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# 2. 더미 데이터 생성 (작은 크기로)
echo ""
echo "🎲 2단계: 더미 데이터 생성 (500개 기업, 빠른 테스트)..."
python generate_phase2_dummy_data.py --n_firms 500 --density 0.03

# 3. Phase 2 실행 (에폭 수 줄임)
echo ""
echo "🚀 3단계: Phase 2 실행 (10 epochs, 빠른 테스트)..."

# Config 수정: EPOCHS를 임시로 10으로 변경
python -c "
import sys
sys.path.insert(0, '.')
from main_phase2_fixed import Config
Config.EPOCHS = 10
Config.EASY_EPOCHS = 7
Config.MEDIUM_EPOCHS = 2
Config.HARD_EPOCHS = 1
Config.FINAL_EPOCHS = 0

# main 실행
from main_phase2_fixed import main
main()
" || python main_phase2_fixed.py

# 4. 결과 확인
echo ""
echo "✅ 4단계: 결과 확인..."

if [ -f "data/processed/node_embeddings_static.pt" ]; then
    echo "   ✓ node_embeddings_static.pt 생성 완료"
    
    # Python으로 임베딩 shape 확인
    python -c "
import torch
emb = torch.load('data/processed/node_embeddings_static.pt')
print(f'   임베딩 Shape: {emb.shape}')
print(f'   임베딩 Norm: {emb.norm(dim=1).mean():.4f}')
    "
else
    echo "   ✗ node_embeddings_static.pt 생성 실패"
fi

if [ -f "data/processed/train_edges.npy" ]; then
    echo "   ✓ train_edges.npy 생성 완료"
    
    python -c "
import numpy as np
train_edges = np.load('data/processed/train_edges.npy')
test_edges = np.load('data/processed/test_edges.npy')
print(f'   Train 엣지: {train_edges.shape[1]:,}')
print(f'   Test 엣지: {test_edges.shape[1]:,}')
    "
else
    echo "   ✗ train_edges.npy 생성 실패"
fi

if [ -f "data/processed/X_feature_matrix.npy" ]; then
    echo "   ✓ X_feature_matrix.npy 생성 완료"
    
    python -c "
import numpy as np
X = np.load('data/processed/X_feature_matrix.npy')
print(f'   피처 Shape: {X.shape}')
print(f'   피처 범위: [{X.min():.2f}, {X.max():.2f}]')
    "
else
    echo "   ✗ X_feature_matrix.npy 생성 실패"
fi

echo ""
echo "=========================================="
echo "테스트 완료! 🎉"
echo "=========================================="
echo ""
echo "생성된 파일:"
echo "  data/processed/"
echo "  ├── disentangled_recipes.pkl       # Phase 1 출력"
echo "  ├── B_matrix.npy                   # Phase 1 출력"
echo "  ├── X_feature_matrix.npy           # Phase 2 피처"
echo "  ├── node_embeddings_static.pt      # Phase 2 임베딩 ⭐"
echo "  ├── train_edges.npy                # Phase 2 Train 엣지"
echo "  ├── test_edges.npy                 # Phase 2 Test 엣지"
echo "  └── tis_score_normalized.npy       # Phase 2 TIS"
echo ""
echo "다음 명령어로 전체 실행 가능:"
echo "  python main_phase2_fixed.py  # 60 epochs 전체"
echo ""
