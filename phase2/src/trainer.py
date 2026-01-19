"""
Curriculum Learning Trainer
============================
커리큘럼 학습을 지원하는 GraphSAGE 트레이너
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurriculumTrainer:
    """
    커리큘럼 학습 트레이너
    
    Parameters
    ----------
    model : torch.nn.Module
        GraphSAGE 모델
    loss_fn : torch.nn.Module
        손실 함수
    optimizer : torch.optim.Optimizer
        옵티마이저
    device : str
        'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
        tis_scores: Optional[torch.Tensor] = None,
        batch_size: int = 1024
    ) -> dict:
        """
        1 에폭 학습
        
        Parameters
        ----------
        x : torch.Tensor, shape (N, D)
            노드 피처
        edge_index : torch.Tensor, shape (2, E)
            전체 그래프 엣지 (message passing용)
        pos_edges : torch.Tensor, shape (2, P)
            Positive 엣지
        neg_edges : torch.Tensor, shape (2, N)
            Negative 엣지
        tis_scores : torch.Tensor, shape (N,), optional
            노드별 TIS 점수
        batch_size : int
            배치 크기
        
        Returns
        -------
        metrics : dict
            - loss: 평균 손실
            - pos_score: Positive 평균 점수
            - neg_score: Negative 평균 점수
        """
        self.model.train()
        
        # 데이터를 device로 이동
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        pos_edges = pos_edges.to(self.device)
        neg_edges = neg_edges.to(self.device)
        
        if tis_scores is not None:
            tis_scores = tis_scores.to(self.device)
        
        total_loss = 0
        num_batches = 0
        all_pos_scores = []
        all_neg_scores = []
        
        # 엣지를 배치로 분할
        num_pos = pos_edges.shape[1]
        num_neg = neg_edges.shape[1]
        
        indices = np.arange(max(num_pos, num_neg))
        np.random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            
            # Positive 배치
            pos_batch_idx = batch_indices % num_pos
            pos_batch = pos_edges[:, pos_batch_idx]
            
            # Negative 배치
            neg_batch_idx = batch_indices % num_neg
            neg_batch = neg_edges[:, neg_batch_idx]
            
            # Forward pass (전체 그래프로 임베딩 생성)
            self.optimizer.zero_grad()
            embeddings = self.model(x, edge_index)
            
            # Positive 예측
            pos_pred = self.model.predict_link(embeddings, pos_batch)
            
            # Negative 예측
            neg_pred = self.model.predict_link(embeddings, neg_batch)
            
            # 레이블 생성
            pos_labels = torch.ones_like(pos_pred)
            neg_labels = torch.zeros_like(neg_pred)
            
            # 예측 및 레이블 결합
            pred = torch.cat([pos_pred, neg_pred])
            labels = torch.cat([pos_labels, neg_labels])
            
            # TIS 점수 (Positive만)
            if tis_scores is not None:
                # Positive 엣지의 도착 노드 TIS
                pos_dst_tis = tis_scores[pos_batch[1]]
                batch_tis = torch.cat([pos_dst_tis, torch.zeros_like(neg_pred)])
            else:
                batch_tis = None
            
            # 손실 계산
            loss = self.loss_fn(pred, labels, batch_tis)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # 통계
            total_loss += loss.item()
            num_batches += 1
            all_pos_scores.append(pos_pred.detach().cpu().numpy())
            all_neg_scores.append(neg_pred.detach().cpu().numpy())
        
        # 평균 계산
        avg_loss = total_loss / num_batches
        avg_pos_score = np.concatenate(all_pos_scores).mean()
        avg_neg_score = np.concatenate(all_neg_scores).mean()
        
        return {
            'loss': avg_loss,
            'avg_pos_score': avg_pos_score,
            'avg_neg_score': avg_neg_score
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
        batch_size: int = 2048
    ) -> dict:
        """
        평가
        
        Returns
        -------
        metrics : dict
            - loss: 평균 손실
            - accuracy: 정확도
            - pos_score: Positive 평균 점수
            - neg_score: Negative 평균 점수
        """
        self.model.eval()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        pos_edges = pos_edges.to(self.device)
        neg_edges = neg_edges.to(self.device)
        
        # 임베딩 생성
        embeddings = self.model(x, edge_index)
        
        # 배치 평가
        pos_scores = []
        neg_scores = []
        
        for start in range(0, pos_edges.shape[1], batch_size):
            end = min(start + batch_size, pos_edges.shape[1])
            batch_pos = pos_edges[:, start:end]
            pred = self.model.predict_link(embeddings, batch_pos)
            pos_scores.append(pred.cpu().numpy())
        
        for start in range(0, neg_edges.shape[1], batch_size):
            end = min(start + batch_size, neg_edges.shape[1])
            batch_neg = neg_edges[:, start:end]
            pred = self.model.predict_link(embeddings, batch_neg)
            neg_scores.append(pred.cpu().numpy())
        
        pos_scores = np.concatenate(pos_scores)
        neg_scores = np.concatenate(neg_scores)
        
        # 정확도 계산
        pos_correct = (pos_scores > 0.5).sum()
        neg_correct = (neg_scores <= 0.5).sum()
        accuracy = (pos_correct + neg_correct) / (len(pos_scores) + len(neg_scores))
        
        return {
            'accuracy': accuracy,
            'avg_pos_score': pos_scores.mean(),
            'avg_neg_score': neg_scores.mean()
        }
    
    def train(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        train_pos_edges: torch.Tensor,
        val_pos_edges: torch.Tensor,
        sampler,
        epochs: int = 20,
        batch_size: int = 1024,
        tis_scores: Optional[torch.Tensor] = None,
        val_ratio: float = 0.2
    ):
        """
        전체 학습 루프 (커리큘럼)
        
        Parameters
        ----------
        sampler : CurriculumNegativeSampler
            네거티브 샘플러
        """
        logger.info("=" * 70)
        logger.info("🚀 커리큘럼 학습 시작")
        logger.info("=" * 70)
        
        for epoch in range(1, epochs + 1):
            # 네거티브 샘플링 (커리큘럼)
            num_train_pos = train_pos_edges.shape[1]
            train_neg_edges, _ = sampler.sample(
                num_samples=num_train_pos * 2,  # 1:2 비율 (학습 속도 향상)
                epoch=epoch,
                total_epochs=epochs
            )
            train_neg_edges = train_neg_edges.to(self.device)
            
            # 학습
            train_metrics = self.train_epoch(
                x, edge_index, train_pos_edges, train_neg_edges,
                tis_scores=tis_scores, batch_size=batch_size
            )
            
            # 검증 (간단히 랜덤 네거티브)
            val_neg_edges, _ = sampler.sample(
                num_samples=val_pos_edges.shape[1],
                epoch=1,  # Random only
                total_epochs=epochs
            )
            val_metrics = self.evaluate(
                x, edge_index, val_pos_edges, val_neg_edges, batch_size=batch_size
            )
            
            # 로깅
            logger.info(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Pos: {train_metrics['avg_pos_score']:.3f} | "
                f"Neg: {train_metrics['avg_neg_score']:.3f} | "
                f"Val Acc: {val_metrics['accuracy']:.3f}"
            )
            
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['accuracy'])
        
        logger.info("=" * 70)
        logger.info("✅ 학습 완료!")
        logger.info("=" * 70)


if __name__ == "__main__":
    print("Trainer 모듈 로드 완료")
