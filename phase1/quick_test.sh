#!/bin/bash
# Phase 1 빠른 테스트 스크립트

echo "=========================================="
echo "Phase 1: Quick Test"
echo "=========================================="

# 1. 패키지 설치
echo ""
echo "📦 1단계: 필수 패키지 설치..."
pip install -q numpy pandas scipy matplotlib

# 2. 더미 데이터 생성
echo ""
echo "🎲 2단계: 더미 데이터 생성 (1,000개 기업)..."
python generate_dummy_data.py --n_firms 1000 --density 0.02

# 3. Phase 1 실행
echo ""
echo "🚀 3단계: Phase 1 실행..."
python main_phase1.py

# 4. 결과 확인
echo ""
echo "✅ 4단계: 결과 확인..."
if [ -f "../data/processed/disentangled_recipes.pkl" ]; then
    echo "   ✓ disentangled_recipes.pkl 생성 완료"
else
    echo "   ✗ disentangled_recipes.pkl 생성 실패"
fi

if [ -f "data/processed/recipes_dataframe.csv" ]; then
    echo "   ✓ recipes_dataframe.csv 생성 완료"
    echo ""
    echo "   CSV 미리보기:"
    head -n 5 data/processed/recipes_dataframe.csv
else
    echo "   ✗ recipes_dataframe.csv 생성 실패"
fi

echo ""
echo "=========================================="
echo "테스트 완료! 🎉"
echo "=========================================="
echo ""
echo "다음 명령어로 상세 분석 가능:"
echo "  python src/check_recipe.py data/processed/disentangled_recipes.pkl"
echo "  python src/debug_deep_dive.py data/processed/disentangled_recipes.pkl --random 3"
echo ""
