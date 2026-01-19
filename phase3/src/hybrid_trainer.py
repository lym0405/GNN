"""
Phase 3 Trainer: Two-Track Hybrid Learning
==========================================
Track A (TGN) + Track B (GraphSEAL) 통합 학습
TIS-aware Soft Label + Ranking Loss 적용
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, List
import logging

from .loss import HybridLoss

logger = logging.getLogger(__name__)


class HybridTrainer:
    """
    Hybrid 모델 학습 (TGN + GraphSEAL)
    TIS-aware BCE + Ranking Loss 사용
    """
    
    def __init__(
        self,
        hybrid_model: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        loss_alpha: float = 0.3,
        soft_negative: float = 0.0,
        ranking_weight: float = 0.1
    ):
        self.model = hybrid_model.to(device)
        self.optimizer = optimizer
        self.device = device
        
        # Hybrid Loss (TIS-aware BCE + Ranking Loss)
        self.criterion = HybridLoss(
            alpha=loss_alpha,
            soft_negative=soft_negative,
            ranking_margin=0.5,
            ranking_weight=ranking_weight
        )
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        self.val_recalls = []
        
        self.best_val_recall = 0.0
        self.patience_counter = 0
    
    def train_epoch(
        self,
        events: List,
        node_features: torch.Tensor,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        tis_scores: Optional[torch.Tensor] = None,
        batch_size: int = 1024
    ) -> float:
        """
        1 에폭 학습
        
        Parameters
        ----------
        events : List of (timestamp, src, dst, edge_feat, label)
        node_features : [N, node_dim]
        node_embeddings : [N, emb_dim] (Phase 2 출력)
        edge_index : [2, E] (전체 그래프)
        tis_scores : [N] (노드별 TIS)
        batch_size : int
        
        Returns
        -------
        avg_loss : float
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # 배치 단위로 처리
        for i in range(0, len(events), batch_size):
            batch_events = events[i:i+batch_size]
            
            # 배치 데이터 추출
            timestamps = torch.tensor([e[0] for e in batch_events], dtype=torch.long)
            src_nodes = torch.tensor([e[1] for e in batch_events], dtype=torch.long)
            dst_nodes = torch.tensor([e[2] for e in batch_events], dtype=torch.long)
            edge_feats = torch.stack([torch.tensor(e[3]) for e in batch_events])
            labels = torch.tensor([e[4] for e in batch_events], dtype=torch.float32)
            
            # GPU로 이동
            timestamps = timestamps.to(self.device)
            src_nodes = src_nodes.to(self.device)
            dst_nodes = dst_nodes.to(self.device)
            edge_feats = edge_feats.to(self.device)
            labels = labels.to(self.device)
            
            # 노드 피처
            src_features = node_features[src_nodes].to(self.device)
            dst_features = node_features[dst_nodes].to(self.device)
            
            # Forward
            logits, outputs = self.model(
                src_nodes=src_nodes,
                dst_nodes=dst_nodes,
                src_features=src_features,
                dst_features=dst_features,
                node_embeddings=node_embeddings.to(self.device),
                edge_index=edge_index.to(self.device),
                timestamps=timestamps,
                tis_scores=None  # TIS는 loss에서 사용
            )
            
            # Loss (TIS-aware Soft Label + Ranking Loss)
            batch_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            batch_tis = tis_scores.to(self.device) if tis_scores is not None else None
            
            total_loss_val, bce_loss, ranking_loss = self.criterion(
                logits=logits,
                labels=labels,
                tis_scores=batch_tis,
                edge_index=batch_edge_index
            )
            
            loss = total_loss_val
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # TGN 메모리 업데이트 (Track A)
            with torch.no_grad():
                self.model.tgn.update_memory_with_batch(
                    src_nodes, dst_nodes, edge_feats, timestamps
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(
        self,
        events: List,
        node_features: torch.Tensor,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        tis_scores: Optional[torch.Tensor] = None,
        k_list: List[int] = [10, 50, 100],
        batch_size: int = 2048
    ) -> Dict:
        """
        평가 (Recall@K 중심)
        
        Returns
        -------
        metrics : Dict
            {
                'loss': float,
                'recall@10': float,
                'recall@50': float,
                ...
            }
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_scores = []
        all_labels = []
        
        # 배치 단위로 처리
        for i in range(0, len(events), batch_size):
            batch_events = events[i:i+batch_size]
            
            # 배치 데이터 추출
            timestamps = torch.tensor([e[0] for e in batch_events], dtype=torch.long)
            src_nodes = torch.tensor([e[1] for e in batch_events], dtype=torch.long)
            dst_nodes = torch.tensor([e[2] for e in batch_events], dtype=torch.long)
            labels = torch.tensor([e[4] for e in batch_events], dtype=torch.float32)
            
            # GPU로 이동
            timestamps = timestamps.to(self.device)
            src_nodes = src_nodes.to(self.device)
            dst_nodes = dst_nodes.to(self.device)
            labels = labels.to(self.device)
            
            # 노드 피처
            src_features = node_features[src_nodes].to(self.device)
            dst_features = node_features[dst_nodes].to(self.device)
            
            # Forward
            logits, outputs = self.model(
                src_nodes=src_nodes,
                dst_nodes=dst_nodes,
                src_features=src_features,
                dst_features=dst_features,
                node_embeddings=node_embeddings.to(self.device),
                edge_index=edge_index.to(self.device),
                timestamps=timestamps,
                tis_scores=None
            )
            
            # Loss (TIS-aware)
            batch_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            batch_tis = tis_scores.to(self.device) if tis_scores is not None else None
            
            total_loss_val, bce_loss, ranking_loss = self.criterion(
                logits=logits,
                labels=labels,
                tis_scores=batch_tis,
                edge_index=batch_edge_index
            )
            
            total_loss += total_loss_val.item()
            num_batches += 1
            
            # 점수 저장
            scores = torch.sigmoid(logits)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        # 전체 데이터 결합
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        
        # Recall@K 계산
        metrics = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        
        for k in k_list:
            recall_k = self._compute_recall_at_k(all_scores, all_labels, k)
            metrics[f'recall@{k}'] = recall_k
        
        return metrics
    
    def _compute_recall_at_k(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """
        Recall@K 계산
        
        상위 K개 예측 중 실제 Positive가 몇 개 포함되는가?
        """
        # 상위 K개 인덱스
        top_k_indices = np.argsort(scores)[-k:]
        
        # Positive 개수
        num_positives = labels.sum()
        
        if num_positives == 0:
            return 0.0
        
        # 상위 K개 중 Positive 개수
        num_hits = labels[top_k_indices].sum()
        
        # Recall
        recall = num_hits / num_positives
        
        return recall
    
    def train(
        self,
        train_events: List,
        val_events: List,
        node_features: torch.Tensor,
        node_embeddings: torch.Tensor,
        train_edge_index: torch.Tensor,
        tis_scores: Optional[torch.Tensor] = None,
        epochs: int = 50,
        batch_size: int = 1024,
        early_stopping_patience: int = 10,
        k_list: List[int] = [10, 50, 100],
        verbose: bool = True
    ):
        """
        전체 학습 루프
        """
        logger.info("=" * 70)
        logger.info("🚀 Hybrid Training 시작 (TGN + GraphSEAL)")
        logger.info("=" * 70)
        
        for epoch in range(1, epochs + 1):
            # TGN 메모리 초기화 (매 에폭마다)
            self.model.tgn.reset_memory()
            
            # 학습
            train_loss = self.train_epoch(
                train_events,
                node_features,
                node_embeddings,
                train_edge_index,
                tis_scores,
                batch_size
            )
            
            # 검증
            val_metrics = self.evaluate(
                val_events,
                node_features,
                node_embeddings,
                train_edge_index,
                tis_scores,
                k_list,
                batch_size * 2
            )
            
            # 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_recalls.append(val_metrics['recall@50'])
            
            # Early Stopping
            if val_metrics['recall@50'] > self.best_val_recall:
                self.best_val_recall = val_metrics['recall@50']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 출력
            if verbose:
                logger.info(
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Recall@10: {val_metrics['recall@10']:.4f} | "
                    f"Recall@50: {val_metrics['recall@50']:.4f} | "
                    f"Recall@100: {val_metrics['recall@100']:.4f}"
                )
            
            # Early Stopping 체크
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"\n⚠️  Early Stopping at Epoch {epoch}")
                break
        
        logger.info("=" * 70)
        logger.info(f"✅ 학습 완료! Best Recall@50: {self.best_val_recall:.4f}")
        logger.info("=" * 70)
