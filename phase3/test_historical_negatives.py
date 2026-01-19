"""
Test script to verify Historical Negatives are loaded correctly
"""
import sys
import torch
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from negative_sampler import Phase3NegativeSampler

def test_historical_negatives():
    """Test that historical negatives are loaded"""
    
    print("=" * 60)
    print("Testing Historical Negatives Loading")
    print("=" * 60)
    
    # Load current network (2024)
    data_dir = Path(__file__).parent.parent / "data"
    current_network = data_dir / "raw" / "posco_network_capital_consumergoods_removed_2024.csv"
    
    if not current_network.exists():
        print(f"❌ Current network file not found: {current_network}")
        # Try alternative name
        current_network = data_dir / "raw" / "posco_network_2024.csv"
        if not current_network.exists():
            print(f"❌ Alternative network file not found: {current_network}")
            print("\n⚠️  Using dummy data for test...")
            # Create dummy current edges
            current_edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
            num_nodes = 100
        else:
            print(f"✓ Using: {current_network.name}")
            df = pd.read_csv(current_network)
            print(f"  Loaded {len(df)} edges")
            # For simplicity, just create dummy edges
            current_edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
            num_nodes = 100
    else:
        print(f"✓ Found: {current_network.name}")
        df = pd.read_csv(current_network)
        print(f"  Loaded {len(df)} edges")
        # For simplicity, just create dummy edges
        current_edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        num_nodes = 100
    
    # Initialize sampler
    print("\n" + "=" * 60)
    print("Initializing Phase3NegativeSampler...")
    print("=" * 60)
    
    sampler = Phase3NegativeSampler(
        num_nodes=num_nodes,
        current_edges=current_edges,
        data_dir=str(data_dir)
    )
    
    # Check results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"✓ Historical Negatives Loaded: {len(sampler.historical_negatives):,}")
    print(f"✓ Positive Edges: {len(sampler.positive_set):,}")
    
    if len(sampler.historical_negatives) > 0:
        print("\n✅ SUCCESS! Historical negatives are being loaded correctly.")
        print(f"\nSample historical negatives (first 5):")
        for i, (src, dst) in enumerate(list(sampler.historical_negatives)[:5]):
            print(f"  {i+1}. ({src}, {dst})")
    else:
        print("\n❌ FAILED! No historical negatives loaded.")
        print("\nPossible issues:")
        print("  1. Check if firm_to_idx_model2.csv has correct column names")
        print("  2. Check if network files for 2020-2023 exist")
        print("  3. Check if firm IDs match between network and mapping files")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_historical_negatives()
