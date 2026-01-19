"""
Link Predictor: MLP-based Link Prediction Classifier
=====================================================
임베딩 기반 링크 예측 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLinkPredictor(nn.Module):
    """
    MLP 기반 링크 예측 분류기
    
    입력: (u_emb, v_emb) -> Concatenation or Hadamard Product
    출력: 링크 존재 확률 (0~1)
    
    Parameters
    ----------
    input_dim : int
        입력 임베딩 차원
    hidden_dims : list of int
        은닉층 차원 리스트
    dropout : float
        드롭아웃 비율
    aggregation : str
        임베딩 결합 방법 ('concat', 'hadamard', 'mean', 'abs_diff')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout: float = 0.3,
        aggregation: str = 'concat'
    ):
        super().__init__()
        
        self.aggregation = aggregation
        
        # 입력 차원 계산
        if aggregation == 'concat':
            agg_dim = input_dim * 2
        elif aggregation in ['hadamard', 'mean', 'abs_diff']:
            agg_dim = input_dim
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # MLP 레이어 구성
        layers = []
        prev_dim = agg_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 출력 레이어
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def aggregate_embeddings(
        self,
        u_emb: torch.Tensor,
        v_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        임베딩 결합
        
        Parameters
        ----------
        u_emb : torch.Tensor [N, D]
        v_emb : torch.Tensor [N, D]
        
        Returns
        -------
        agg_emb : torch.Tensor [N, D or 2D]
        """
        if self.aggregation == 'concat':
            return torch.cat([u_emb, v_emb], dim=1)
        elif self.aggregation == 'hadamard':
            return u_emb * v_emb
        elif self.aggregation == 'mean':
            return (u_emb + v_emb) / 2
        elif self.aggregation == 'abs_diff':
            return torch.abs(u_emb - v_emb)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def forward(
        self,
        u_emb: torch.Tensor,
        v_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        순전파
        
        Parameters
        ----------
        u_emb : torch.Tensor [N, D]
            소스 노드 임베딩
        v_emb : torch.Tensor [N, D]
            타겟 노드 임베딩
        
        Returns
        -------
        logits : torch.Tensor [N, 1]
            링크 존재 로짓
        """
        # 임베딩 결합
        agg_emb = self.aggregate_embeddings(u_emb, v_emb)
        
        # MLP 통과
        logits = self.mlp(agg_emb)
        
        return logits
    
    def predict(
        self,
        u_emb: torch.Tensor,
        v_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        예측 (확률)
        
        Returns
        -------
        probs : torch.Tensor [N]
            링크 존재 확률
        """
        logits = self.forward(u_emb, v_emb)
        probs = torch.sigmoid(logits).squeeze()
        return probs


class EnsembleLinkPredictor(nn.Module):
    """
    앙상블 링크 예측기
    
    여러 aggregation 방법을 결합
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout: float = 0.3,
        aggregations: list = ['concat', 'hadamard']
    ):
        super().__init__()
        
        self.predictors = nn.ModuleList([
            MLPLinkPredictor(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                aggregation=agg
            )
            for agg in aggregations
        ])
        
        # 앙상블 가중치
        self.ensemble_weights = nn.Parameter(
            torch.ones(len(aggregations)) / len(aggregations)
        )
    
    def forward(
        self,
        u_emb: torch.Tensor,
        v_emb: torch.Tensor
    ) -> torch.Tensor:
        """앙상블 예측"""
        logits_list = [
            predictor(u_emb, v_emb)
            for predictor in self.predictors
        ]
        
        # 가중 평균
        logits = torch.stack(logits_list, dim=0)  # [K, N, 1]
        weights = F.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        
        ensemble_logits = (logits * weights).sum(dim=0)
        
        return ensemble_logits
    
    def predict(
        self,
        u_emb: torch.Tensor,
        v_emb: torch.Tensor
    ) -> torch.Tensor:
        """예측 확률"""
        logits = self.forward(u_emb, v_emb)
        probs = torch.sigmoid(logits).squeeze()
        return probs
