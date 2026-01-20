"""
Phase 3: GraphSEAL Fast Training Script
========================================
GraphSEAL만 사용하여 빠르게 학습 (30 epochs)
모델 규모는 키워서 성능 유지

Usage:
    python phase3/train_graphseal_fast.py
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import sparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3.src.graphseal import GraphSEAL


class Config:
    """Configuration for fast GraphSEAL training."""
    
    # Data paths
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DIR = PROJECT_ROOT / "phase3" / "output" / "graphseal_fast"
    
    # Input files
    H_MATRIX = DATA_RAW / "H_csr_model2.npz"
    FIRM_TO_IDX = DATA_RAW / "firm_to_idx_model2.csv"
    TRAIN_EDGES = DATA_PROCESSED / "train_edges.npy"
    TEST_EDGES = DATA_PROCESSED / "test_edges.npy"
    NODE_EMBEDDINGS = DATA_PROCESSED / "node_embeddings_static.pt"
    
    # Model parameters (Larger model)
    EMBEDDING_DIM = 256  # Increased from 128
    HIDDEN_DIM = 512     # Increased from 256
    NUM_LAYERS = 4       # Increased from 3
    NUM_HOPS = 3         # Increased from 2
    DROPOUT = 0.2
    
    # Training parameters
    EPOCHS = 30
    BATCH_SIZE = 256     # Larger batch for speed
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Negative sampling
    NEG_SAMPLES_PER_POS = 5
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_logging(config: Config):
    """Setup logging."""
    log_dir = config.OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"GraphSEAL Fast Training - Log file: {log_file}")
    return logger


def load_data(config: Config, logger: logging.Logger):
    """Load all required data."""
    logger.info("Loading data...")
    
    # Load H matrix (supply network)
    H = sparse.load_npz(config.H_MATRIX)
    logger.info(f"Loaded H matrix: shape {H.shape}, {H.nnz} edges")
    
    # Convert to edge_index
    row, col = H.nonzero()
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    logger.info(f"Edge index: {edge_index.shape}")
    
    # Load node embeddings
    node_embeddings = torch.load(config.NODE_EMBEDDINGS)
    logger.info(f"Loaded node embeddings: {node_embeddings.shape}")
    
    # Resize if dimensions don't match
    if node_embeddings.shape[1] != config.EMBEDDING_DIM:
        logger.info(f"Resizing embeddings from {node_embeddings.shape[1]} to {config.EMBEDDING_DIM}")
        resize_layer = nn.Linear(node_embeddings.shape[1], config.EMBEDDING_DIM)
        with torch.no_grad():
            node_embeddings = resize_layer(node_embeddings)
    
    # Load train/test edges
    train_edges = np.load(config.TRAIN_EDGES)
    test_edges = np.load(config.TEST_EDGES)
    logger.info(f"Train edges: {len(train_edges)}, Test edges: {len(test_edges)}")
    
    return {
        'H': H,
        'edge_index': edge_index,
        'node_embeddings': node_embeddings,
        'train_edges': train_edges,
        'test_edges': test_edges,
        'num_nodes': H.shape[0]
    }


def negative_sampling(
    pos_edges: np.ndarray,
    num_nodes: int,
    num_neg_per_pos: int,
    existing_edges_set: set,
    logger: logging.Logger
) -> np.ndarray:
    """Generate negative samples."""
    logger.info(f"Generating {num_neg_per_pos} negative samples per positive edge...")
    
    neg_edges = []
    num_pos = len(pos_edges)
    
    for i, (src, dst) in enumerate(pos_edges):
        if i % 1000 == 0:
            logger.info(f"  Progress: {i}/{num_pos}")
        
        count = 0
        attempts = 0
        max_attempts = num_neg_per_pos * 10
        
        while count < num_neg_per_pos and attempts < max_attempts:
            # Random negative edge
            neg_src = np.random.randint(0, num_nodes)
            neg_dst = np.random.randint(0, num_nodes)
            
            # Check if edge doesn't exist
            if (neg_src, neg_dst) not in existing_edges_set and neg_src != neg_dst:
                neg_edges.append([neg_src, neg_dst])
                count += 1
            
            attempts += 1
    
    neg_edges = np.array(neg_edges)
    logger.info(f"Generated {len(neg_edges)} negative samples")
    
    return neg_edges


def create_dataloaders(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    test_pos: np.ndarray,
    test_neg: np.ndarray,
    batch_size: int,
    logger: logging.Logger
):
    """Create PyTorch dataloaders."""
    logger.info("Creating dataloaders...")
    
    # Combine positive and negative samples
    train_edges = np.vstack([train_pos, train_neg])
    train_labels = np.hstack([
        np.ones(len(train_pos)),
        np.zeros(len(train_neg))
    ])
    
    test_edges = np.vstack([test_pos, test_neg])
    test_labels = np.hstack([
        np.ones(len(test_pos)),
        np.zeros(len(test_neg))
    ])
    
    # Convert to tensors
    train_edges = torch.tensor(train_edges, dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    test_edges = torch.tensor(test_edges, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.float)
    
    # Create datasets
    train_dataset = TensorDataset(train_edges, train_labels)
    test_dataset = TensorDataset(test_edges, test_labels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    edge_index: torch.Tensor,
    node_embeddings: torch.Tensor,
    device: str,
    logger: logging.Logger
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_edges, batch_labels in train_loader:
        batch_edges = batch_edges.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(
            node_emb=node_embeddings,
            edge_index=edge_index,
            query_edges=batch_edges
        )
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    node_embeddings: torch.Tensor,
    device: str,
    logger: logging.Logger
) -> dict:
    """Evaluate the model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch_edges, batch_labels in test_loader:
        batch_edges = batch_edges.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        logits = model(
            node_emb=node_embeddings,
            edge_index=edge_index,
            query_edges=batch_edges
        )
        
        # Loss
        loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
        total_loss += loss.item()
        
        # Predictions
        preds = torch.sigmoid(logits)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch_labels.cpu().numpy())
        
        num_batches += 1
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    
    binary_preds = (all_preds > 0.5).astype(int)
    
    metrics = {
        'loss': total_loss / num_batches,
        'auc': roc_auc_score(all_labels, all_preds),
        'ap': average_precision_score(all_labels, all_preds),
        'accuracy': accuracy_score(all_labels, binary_preds)
    }
    
    return metrics


def save_results(
    model: nn.Module,
    node_embeddings: torch.Tensor,
    config: Config,
    logger: logging.Logger
):
    """Save trained model and results."""
    logger.info("Saving results...")
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = config.OUTPUT_DIR / "graphseal_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save final node embeddings (from model)
    emb_path = config.OUTPUT_DIR / "node_embeddings_graphseal.pt"
    torch.save(node_embeddings, emb_path)
    logger.info(f"Saved embeddings to {emb_path}")
    
    logger.info("All results saved successfully")


def main():
    """Main training function."""
    config = Config()
    logger = setup_logging(config)
    
    try:
        logger.info("=" * 80)
        logger.info("GraphSEAL Fast Training - Starting")
        logger.info("=" * 80)
        logger.info(f"Device: {config.DEVICE}")
        logger.info(f"Epochs: {config.EPOCHS}")
        logger.info(f"Batch size: {config.BATCH_SIZE}")
        logger.info(f"Model: Embedding={config.EMBEDDING_DIM}, Hidden={config.HIDDEN_DIM}, Layers={config.NUM_LAYERS}")
        
        # Load data
        data = load_data(config, logger)
        
        # Move to device
        device = torch.device(config.DEVICE)
        edge_index = data['edge_index'].to(device)
        node_embeddings = data['node_embeddings'].to(device)
        
        # Create existing edges set for negative sampling
        existing_edges = set(map(tuple, data['train_edges']))
        existing_edges.update(map(tuple, data['test_edges']))
        logger.info(f"Total existing edges: {len(existing_edges)}")
        
        # Generate negative samples
        train_neg = negative_sampling(
            data['train_edges'],
            data['num_nodes'],
            config.NEG_SAMPLES_PER_POS,
            existing_edges,
            logger
        )
        
        test_neg = negative_sampling(
            data['test_edges'],
            data['num_nodes'],
            config.NEG_SAMPLES_PER_POS,
            existing_edges,
            logger
        )
        
        # Create dataloaders
        train_loader, test_loader = create_dataloaders(
            data['train_edges'],
            train_neg,
            data['test_edges'],
            test_neg,
            config.BATCH_SIZE,
            logger
        )
        
        # Initialize model
        logger.info("Initializing GraphSEAL model...")
        model = GraphSEAL(
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            num_hops=config.NUM_HOPS,
            dropout=config.DROPOUT
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Training loop
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)
        
        best_auc = 0.0
        best_epoch = 0
        
        for epoch in range(config.EPOCHS):
            epoch_start = time.time()
            
            # Train
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                edge_index,
                node_embeddings,
                device,
                logger
            )
            
            # Evaluate
            test_metrics = evaluate(
                model,
                test_loader,
                edge_index,
                node_embeddings,
                device,
                logger
            )
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{config.EPOCHS} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_metrics['loss']:.4f} | "
                f"AUC: {test_metrics['auc']:.4f} | "
                f"AP: {test_metrics['ap']:.4f} | "
                f"Acc: {test_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if test_metrics['auc'] > best_auc:
                best_auc = test_metrics['auc']
                best_epoch = epoch + 1
                logger.info(f"  → New best AUC: {best_auc:.4f}")
        
        logger.info("=" * 80)
        logger.info(f"Training completed! Best AUC: {best_auc:.4f} at epoch {best_epoch}")
        logger.info("=" * 80)
        
        # Save results
        save_results(model, node_embeddings, config, logger)
        
        logger.info("GraphSEAL Fast Training - Completed Successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
