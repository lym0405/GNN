"""
GraphSAGE Model for Link Prediction
====================================
2-Layer GraphSAGE with Mean Aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional


class GraphSAGE(nn.Module):
    """
    GraphSAGE for Link Prediction
    
    Parameters
    ----------
    in_dim : int
        입력 피처 차원
    hidden_dim : int
        은닉층 차원
    out_dim : int
        출력 임베딩 차원
    dropout : float
        드롭아웃 비율
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.3
    ):
        super(GraphSAGE, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        # GraphSAGE Layers
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr='mean')
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.drop = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor, shape (N, in_dim)
            노드 피처
        edge_index : torch.Tensor, shape (2, E)
            엣지 인덱스
        
        Returns
        -------
        embeddings : torch.Tensor, shape (N, out_dim)
            노드 임베딩
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        
        # L2 Normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def predict_link(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        링크 예측 (Dot Product)
        
        Parameters
        ----------
        z : torch.Tensor, shape (N, out_dim)
            노드 임베딩
        edge_index : torch.Tensor, shape (2, E)
            예측할 엣지 인덱스
        
        Returns
        -------
        scores : torch.Tensor, shape (E,)
            링크 존재 확률 (0~1)
        """
        src_emb = z[edge_index[0]]  # (E, out_dim)
        dst_emb = z[edge_index[1]]  # (E, out_dim)
        
        # Dot Product
        scores = (src_emb * dst_emb).sum(dim=1)  # (E,)
        
        # Sigmoid
        scores = torch.sigmoid(scores)
        
        return scores
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        노드 임베딩 생성 (forward와 동일)
        """
        return self.forward(x, edge_index)


class LinkPredictionHead(nn.Module):
    """
    링크 예측 헤드 (선택적)
    Dot Product 대신 MLP 사용 가능
    """
    
    def __init__(self, emb_dim: int, hidden_dim: int = 32):
        super(LinkPredictionHead, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src_emb = z[edge_index[0]]
        dst_emb = z[edge_index[1]]
        
        # Concatenate
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        
        # MLP
        scores = self.mlp(edge_emb).squeeze()
        
        return scores


if __name__ == "__main__":
    print("=" * 70)
    print("GraphSAGE 모델 테스트")
    print("=" * 70)
    
    # 더미 데이터
    N = 1000  # 노드 수
    E = 5000  # 엣지 수
    in_dim = 73  # 피처 차원
    
    # 랜덤 데이터 생성
    x = torch.randn(N, in_dim)
    edge_index = torch.randint(0, N, (2, E))
    
    # 모델 초기화
    model = GraphSAGE(in_dim=in_dim, hidden_dim=64, out_dim=32, dropout=0.3)
    
    print(f"\n✅ 모델 초기화 완료")
    print(f"   - 입력 차원: {in_dim}")
    print(f"   - 은닉 차원: 64")
    print(f"   - 출력 차원: 32")
    print(f"   - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(x, edge_index)
        print(f"\n✅ Forward Pass")
        print(f"   - 임베딩 shape: {embeddings.shape}")
        print(f"   - 임베딩 norm: {embeddings.norm(dim=1).mean():.4f}")
        
        # Link prediction
        test_edges = torch.randint(0, N, (2, 100))
        scores = model.predict_link(embeddings, test_edges)
        print(f"\n✅ Link Prediction")
        print(f"   - 예측 점수 shape: {scores.shape}")
        print(f"   - 점수 범위: {scores.min():.4f} ~ {scores.max():.4f}")
        print(f"   - 평균 점수: {scores.mean():.4f}")
