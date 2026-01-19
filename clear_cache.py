#!/usr/bin/env python3
"""
Cache Clearing Utility
======================
Phase 2/3의 캐시된 그래프 데이터를 삭제하는 유틸리티

사용법:
    python clear_cache.py              # 모든 캐시 삭제
    python clear_cache.py --phase2     # Phase 2 캐시만 삭제
    python clear_cache.py --phase3     # Phase 3 캐시만 삭제
"""

import argparse
import shutil
from pathlib import Path


def clear_cache(phase: str = "all"):
    """
    캐시 삭제
    
    Parameters
    ----------
    phase : str
        "all", "phase2", "phase3"
    """
    project_root = Path(__file__).parent
    cache_dir = project_root / "data" / "processed" / "cache"
    
    if not cache_dir.exists():
        print("📂 캐시 디렉토리가 없습니다.")
        return
    
    if phase == "all":
        # 전체 캐시 삭제
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("✅ 전체 캐시 삭제 완료")
        
    elif phase == "phase2":
        # Phase 2 캐시만 삭제
        for f in cache_dir.glob("static_*"):
            f.unlink()
            print(f"   🗑️  {f.name}")
        print("✅ Phase 2 캐시 삭제 완료")
        
    elif phase == "phase3":
        # Phase 3 캐시만 삭제
        temporal_cache = cache_dir / "temporal_data.pkl"
        if temporal_cache.exists():
            temporal_cache.unlink()
            print(f"   🗑️  {temporal_cache.name}")
        print("✅ Phase 3 캐시 삭제 완료")
    
    else:
        print(f"❌ 잘못된 phase: {phase}")


def main():
    parser = argparse.ArgumentParser(description="캐시 삭제 유틸리티")
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Phase 2 캐시만 삭제"
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Phase 3 캐시만 삭제"
    )
    
    args = parser.parse_args()
    
    if args.phase2:
        clear_cache("phase2")
    elif args.phase3:
        clear_cache("phase3")
    else:
        clear_cache("all")


if __name__ == "__main__":
    main()
