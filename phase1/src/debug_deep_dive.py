"""
Recipe Deep Dive Analysis
==========================
특정 기업의 레시피를 상세 분석하는 디버깅 도구
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def analyze_firm_recipe(recipe_dict: dict, firm_id: str, sector_names: list = None):
    """
    특정 기업의 레시피 상세 분석
    
    Parameters
    ----------
    recipe_dict : dict
        {firm_id: np.ndarray(33,)} 레시피 딕셔너리
    firm_id : str
        분석할 기업 ID
    sector_names : list, optional
        33개 산업명 리스트
    """
    if firm_id not in recipe_dict:
        print(f"❌ 기업 '{firm_id}'를 찾을 수 없습니다.")
        return
    
    recipe = recipe_dict[firm_id]
    
    if sector_names is None:
        sector_names = [f'Sector_{i:02d}' for i in range(1, 34)]
    
    print("\n" + "=" * 70)
    print(f"🔍 기업 '{firm_id}' 레시피 분석")
    print("=" * 70)
    
    # 기본 통계
    print("\n[기본 통계]")
    print(f"  합계: {recipe.sum():.6f}")
    print(f"  평균: {recipe.mean():.6f}")
    print(f"  표준편차: {recipe.std():.6f}")
    print(f"  최솟값: {recipe.min():.6f}")
    print(f"  최댓값: {recipe.max():.6f}")
    
    # 상위 10개 산업
    print("\n[상위 10개 산업]")
    top10_idx = np.argsort(recipe)[-10:][::-1]
    print(f"{'순위':<5} {'산업명':<20} {'비율':<10} {'누적':<10}")
    print("-" * 50)
    
    cumsum = 0
    for rank, idx in enumerate(top10_idx, 1):
        val = recipe[idx]
        cumsum += val
        print(f"{rank:<5} {sector_names[idx]:<20} {val:.4f}     {cumsum:.4f}")
    
    # 다양성 지표
    print("\n[다양성 지표]")
    active_count = (recipe > 0.01).sum()
    print(f"  활성 산업 수 (>1%): {active_count}/33")
    
    entropy = -(recipe * np.log(recipe + 1e-10)).sum()
    max_entropy = np.log(33)
    print(f"  엔트로피: {entropy:.3f} / {max_entropy:.3f} ({entropy/max_entropy*100:.1f}%)")
    
    gini = calculate_gini(recipe)
    print(f"  Gini 계수: {gini:.3f} (0=균등, 1=불균등)")
    
    # 시각화
    visualize_recipe(recipe, sector_names, firm_id)


def calculate_gini(values: np.ndarray) -> float:
    """Gini 계수 계산"""
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * (sorted_values * index).sum()) / (n * sorted_values.sum()) - (n + 1) / n


def visualize_recipe(recipe: np.ndarray, sector_names: list, firm_id: str):
    """레시피 시각화"""
    try:
        import matplotlib.pyplot as plt
        
        # 상위 15개만 표시
        top15_idx = np.argsort(recipe)[-15:][::-1]
        top15_values = recipe[top15_idx]
        top15_names = [sector_names[i] for i in top15_idx]
        
        plt.figure(figsize=(12, 6))
        
        # 바 차트
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(15), top15_values)
        plt.xticks(range(15), top15_names, rotation=45, ha='right')
        plt.ylabel('비율')
        plt.title(f'상위 15개 산업 ({firm_id})')
        plt.grid(axis='y', alpha=0.3)
        
        # 색상 강조
        for i, bar in enumerate(bars):
            if i < 3:
                bar.set_color('coral')
            else:
                bar.set_color('skyblue')
        
        # 파이 차트
        plt.subplot(1, 2, 2)
        top5_idx = np.argsort(recipe)[-5:][::-1]
        top5_values = recipe[top5_idx]
        other_value = 1 - top5_values.sum()
        
        labels = [sector_names[i] for i in top5_idx] + ['기타']
        values = list(top5_values) + [other_value]
        
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'산업 구성 ({firm_id})')
        
        plt.tight_layout()
        plt.savefig(f'recipe_analysis_{firm_id}.png', dpi=150, bbox_inches='tight')
        print(f"\n💾 그래프 저장: recipe_analysis_{firm_id}.png")
        
        plt.show()
    
    except ImportError:
        print("\n⚠️ matplotlib가 설치되지 않아 시각화를 건너뜁니다.")


def compare_firms(recipe_dict: dict, firm_ids: list, sector_names: list = None):
    """여러 기업의 레시피 비교"""
    if sector_names is None:
        sector_names = [f'Sector_{i:02d}' for i in range(1, 34)]
    
    print("\n" + "=" * 70)
    print(f"📊 기업 비교 분석")
    print("=" * 70)
    
    # 레시피 행렬 구성
    recipes = []
    valid_ids = []
    for fid in firm_ids:
        if fid in recipe_dict:
            recipes.append(recipe_dict[fid])
            valid_ids.append(fid)
        else:
            print(f"⚠️ 기업 '{fid}' 없음")
    
    if len(recipes) < 2:
        print("❌ 비교할 기업이 부족합니다.")
        return
    
    recipes = np.array(recipes)
    
    # 코사인 유사도 계산
    print("\n[코사인 유사도]")
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            cos_sim = np.dot(recipes[i], recipes[j]) / (
                np.linalg.norm(recipes[i]) * np.linalg.norm(recipes[j])
            )
            print(f"  {valid_ids[i]} ↔ {valid_ids[j]}: {cos_sim:.4f}")
    
    # 유클리드 거리
    print("\n[유클리드 거리]")
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            dist = np.linalg.norm(recipes[i] - recipes[j])
            print(f"  {valid_ids[i]} ↔ {valid_ids[j]}: {dist:.4f}")


def main():
    parser = argparse.ArgumentParser(description="레시피 상세 분석 도구")
    parser.add_argument('recipe_path', type=str, help='disentangled_recipes.pkl 경로')
    parser.add_argument('--firm', type=str, help='분석할 기업 ID')
    parser.add_argument('--compare', nargs='+', help='비교할 기업 ID 리스트')
    parser.add_argument('--random', type=int, help='랜덤 샘플링 개수')
    
    args = parser.parse_args()
    
    # 레시피 로드
    with open(args.recipe_path, 'rb') as f:
        recipe_dict = pickle.load(f)
    
    print(f"✅ 레시피 로드: {len(recipe_dict)} 기업")
    
    # 단일 기업 분석
    if args.firm:
        analyze_firm_recipe(recipe_dict, args.firm)
    
    # 기업 비교
    elif args.compare:
        compare_firms(recipe_dict, args.compare)
    
    # 랜덤 샘플링
    elif args.random:
        import random
        sample_ids = random.sample(list(recipe_dict.keys()), min(args.random, len(recipe_dict)))
        print(f"\n🎲 랜덤 샘플링: {sample_ids}")
        for fid in sample_ids:
            analyze_firm_recipe(recipe_dict, fid)
    
    else:
        print("\n사용법:")
        print("  python debug_deep_dive.py <recipe_path> --firm <firm_id>")
        print("  python debug_deep_dive.py <recipe_path> --compare <firm1> <firm2> ...")
        print("  python debug_deep_dive.py <recipe_path> --random <n>")


if __name__ == "__main__":
    main()
