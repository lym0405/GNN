"""
Phase 3 Trainer: Link Prediction Training
==========================================
링크 예측 모델 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LinkPredictionTrainer:
    """
    링크 예측 모델 학습기
    
    Parameters
    ----------
    model : nn.Module
        링크 예측 모델
    optimizer : optim.Optimizer
        옵티마이저
    criterion : nn.Module
        손실 함수
    device : str
        디바이스
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module = None,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        embeddings: torch.Tensor
    ) -> float:
        """
        1 Epoch 학습
        
        Parameters
        ----------
        train_loader : DataLoader
            (u, v, label) 배치
        embeddings : torch.Tensor [N, D]
            노드 임베딩 (고정)
        
        Returns
        -------
        avg_loss : float
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        embeddings = embeddings.to(self.device)
        
        for batch_u, batch_v, batch_y in train_loader:
            batch_u = batch_u.to(self.device)
            batch_v = batch_v.to(self.device)
            batch_y = batch_y.to(self.device).float().view(-1, 1)
            
            # 임베딩 추출
            u_emb = embeddings[batch_u]
            v_emb = embeddings[batch_v]
            
            # 순전파
            logits = self.model(u_emb, v_emb)
            loss = self.criterion(logits, batch_y)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        검증 평가
        
        Returns
        -------
        metrics : dict
            loss, accuracy
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        embeddings = embeddings.to(self.device)
        
        for batch_u, batch_v, batch_y in val_loader:
            batch_u = batch_u.to(self.device)
            batch_v = batch_v.to(self.device)
            batch_y = batch_y.to(self.device).float().view(-1, 1)
            
            # 임베딩 추출
            u_emb = embeddings[batch_u]
            v_emb = embeddings[batch_v]
            
            # 순전파
            logits = self.model(u_emb, v_emb)
            loss = self.criterion(logits, batch_y)
            
            # 예측
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            total_loss += loss.item() * batch_y.size(0)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        embeddings: torch.Tensor,
        epochs: int,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ):
        """
        전체 학습
        
        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        embeddings : torch.Tensor [N, D]
        epochs : int
        early_stopping_patience : int
        verbose : bool
        """
        logger.info("=" * 70)
        logger.info("🚀 Phase 3: 링크 예측 학습 시작")
        logger.info("=" * 70)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            # 학습
            train_loss = self.train_epoch(train_loader, embeddings)
            
            # 검증
            val_metrics = self.evaluate(val_loader, embeddings)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            
            # 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 로깅
            if verbose:
                logger.info(
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"⏹️  Early stopping at epoch {epoch}")
                break
        
        # 최적 모델 복원
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"✅ 최적 모델 복원 (Val Loss: {best_val_loss:.4f})")
        
        logger.info("=" * 70)
    
    @torch.no_grad()
    def predict(
        self,
        edge_index: torch.Tensor,
        embeddings: torch.Tensor,
        batch_size: int = 4096
    ) -> np.ndarray:
        """
        배치 예측
        
        Parameters
        ----------
        edge_index : torch.Tensor [2, E]
        embeddings : torch.Tensor [N, D]
        batch_size : int
        
        Returns
        -------
        scores : np.ndarray [E]
            예측 확률
        """
        self.model.eval()
        embeddings = embeddings.to(self.device)
        
        num_edges = edge_index.shape[1]
        all_scores = []
        
        for i in range(0, num_edges, batch_size):
            batch_edges = edge_index[:, i:i+batch_size].to(self.device)
            
            u_emb = embeddings[batch_edges[0]]
            v_emb = embeddings[batch_edges[1]]
            
            scores = self.model.predict(u_emb, v_emb)
            all_scores.append(scores.cpu().numpy())
        
        return np.concatenate(all_scores)


def prepare_dataloader(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """
    DataLoader 생성
    
    Parameters
    ----------
    edge_index : torch.Tensor [2, E]
    labels : torch.Tensor [E]
    batch_size : int
    shuffle : bool
    
    Returns
    -------
    loader : DataLoader
    """
    u = edge_index[0]
    v = edge_index[1]
    
    dataset = TensorDataset(u, v, labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )
    
    return loader
