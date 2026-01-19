"""
SC-TGN: Supply Chain Temporal Graph Network (Track A)
=====================================================
시계열 동적 예측을 위한 TGN 모델

참고: Temporal Graph Networks (Rossi et al., ICLR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class MemoryModule(nn.Module):
    """
    메모리 모듈: 각 노드의 과거 상호작용 기억
    """
    
    def __init__(self, num_nodes: int, memory_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        
        # 메모리 벡터 (학습되지 않음, 동적 업데이트)
        self.register_buffer(
            'memory',
            torch.zeros(num_nodes, memory_dim)
        )
        
        # 마지막 업데이트 시간
        self.register_buffer(
            'last_update',
            torch.zeros(num_nodes, dtype=torch.long)
        )
    
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """메모리 조회"""
        return self.memory[node_ids]
    
    def update_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """메모리 업데이트"""
        self.memory[node_ids] = messages
        self.last_update[node_ids] = timestamps
    
    def reset(self):
        """메모리 초기화"""
        self.memory.zero_()
        self.last_update.zero_()


class TimeEncoder(nn.Module):
    """
    시간 인코딩: 시간 간격을 피처로 변환
    """
    
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        
        self.w = nn.Linear(1, time_dim)
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        timestamps : torch.Tensor [B]
        
        Returns
        -------
        time_encoding : torch.Tensor [B, time_dim]
        """
        # [B, 1]
        t = timestamps.unsqueeze(-1).float()
        
        # [B, time_dim]
        return torch.cos(self.w(t))


class MessageAggregator(nn.Module):
    """
    메시지 집계: 이웃 노드들로부터 메시지 수집
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        message_dim: int
    ):
        super().__init__()
        
        self.message_dim = message_dim
        
        # 메시지 함수
        self.message_fc = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + time_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
    
    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        edge_feat: torch.Tensor,
        time_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src_memory : [E, node_dim]
        dst_memory : [E, node_dim]
        edge_feat : [E, edge_dim]
        time_encoding : [E, time_dim]
        
        Returns
        -------
        messages : [E, message_dim]
        """
        # 모든 피처 결합
        x = torch.cat([
            src_memory,
            dst_memory,
            edge_feat,
            time_encoding
        ], dim=-1)
        
        # 메시지 생성
        messages = self.message_fc(x)
        
        return messages


class MemoryUpdater(nn.Module):
    """
    메모리 업데이터: GRU 기반 메모리 갱신
    """
    
    def __init__(self, memory_dim: int, message_dim: int):
        super().__init__()
        
        self.gru = nn.GRUCell(message_dim, memory_dim)
    
    def forward(
        self,
        memory: torch.Tensor,
        messages: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        memory : [N, memory_dim]
        messages : [N, message_dim]
        
        Returns
        -------
        updated_memory : [N, memory_dim]
        """
        return self.gru(messages, memory)


class SC_TGN(nn.Module):
    """
    Supply Chain Temporal Graph Network
    
    시계열 거래 네트워크에서 미래 링크 예측
    
    [최적화] 벡터화된 메시지 집계 (Python Loop 제거)
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        edge_dim: int,
        memory_dim: int = 128,
        time_dim: int = 32,
        message_dim: int = 128,
        embedding_dim: int = 64
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        
        # 모듈들
        self.memory = MemoryModule(num_nodes, memory_dim)
        self.time_encoder = TimeEncoder(time_dim)
        self.message_aggregator = MessageAggregator(
            memory_dim, edge_dim, time_dim, message_dim
        )
        self.memory_updater = MemoryUpdater(memory_dim, message_dim)
        
        # 노드 피처 임베딩
        self.node_encoder = nn.Linear(node_dim, memory_dim)
        
        # 최종 임베딩 생성
        self.embedding_layer = nn.Sequential(
            nn.Linear(memory_dim + node_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def reset_memory(self):
        """메모리 초기화"""
        self.memory.reset()
    
    def get_embeddings(
        self,
        node_ids: torch.Tensor,
        node_features: torch.Tensor,
        current_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        노드 임베딩 생성
        
        Parameters
        ----------
        node_ids : torch.Tensor [B]
        node_features : torch.Tensor [B, node_dim]
        current_time : Optional[torch.Tensor] [B]
        
        Returns
        -------
        embeddings : torch.Tensor [B, embedding_dim]
        """
        # 메모리 조회
        memory = self.memory.get_memory(node_ids)  # [B, memory_dim]
        
        # 노드 피처와 메모리 결합
        combined = torch.cat([memory, node_features], dim=-1)
        
        # 임베딩 생성
        embeddings = self.embedding_layer(combined)
        
        return embeddings
    
    def update_memory_with_batch(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """
        배치 단위 메모리 업데이트
        
        [최적화] Python Loop 제거 -> 벡터 연산으로 일괄 처리 (10-100x 빠름)
        
        Parameters
        ----------
        src_nodes : torch.Tensor [E]
        dst_nodes : torch.Tensor [E]
        edge_features : torch.Tensor [E, edge_dim]
        timestamps : torch.Tensor [E]
        """
        # 1. 메시지 생성 (기존과 동일)
        src_memory = self.memory.get_memory(src_nodes)
        dst_memory = self.memory.get_memory(dst_nodes)
        time_encoding = self.time_encoder(timestamps)
        
        # [E, message_dim]
        messages = self.message_aggregator(
            src_memory, dst_memory, edge_features, time_encoding
        )
        
        # 2. [최적화] 메시지 집계 (Loop 제거)
        # 소스/타겟 구분 없이 한 번에 처리
        all_nodes = torch.cat([src_nodes, dst_nodes])
        all_messages = torch.cat([messages, messages])
        
        # 유니크 노드와 역참조 인덱스 추출
        unique_nodes, inverse_indices = torch.unique(all_nodes, return_inverse=True)
        
        n_unique = unique_nodes.size(0)
        msg_dim = messages.size(1)
        device = self.memory.memory.device
        
        # (1) 메시지 합계 계산 (index_add_ 사용)
        aggr_messages = torch.zeros(n_unique, msg_dim, device=device)
        aggr_messages.index_add_(0, inverse_indices, all_messages)
        
        # (2) 노드별 카운트 계산 (평균을 위해)
        counts = torch.zeros(n_unique, 1, device=device)
        ones = torch.ones(all_nodes.size(0), 1, device=device)
        counts.index_add_(0, inverse_indices, ones)
        
        # (3) 평균 메시지
        aggr_messages = aggr_messages / counts.clamp(min=1e-9)
        
        # 3. 메모리 일괄 업데이트 (Batch GRU)
        current_memory = self.memory.get_memory(unique_nodes)
        updated_memory = self.memory_updater(current_memory, aggr_messages)
        
        # 4. 메모리 뱅크 갱신
        # 시간은 배치의 최신 시간으로 일괄 갱신 (속도 우선)
        batch_max_time = timestamps.max().expand(n_unique)
        
        self.memory.update_memory(
            unique_nodes,
            updated_memory,
            batch_max_time
        )
    
    def forward(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        src_features: torch.Tensor,
        dst_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        링크 예측 (logits)
        
        Parameters
        ----------
        src_nodes : torch.Tensor [E]
        dst_nodes : torch.Tensor [E]
        src_features : torch.Tensor [E, node_dim]
        dst_features : torch.Tensor [E, node_dim]
        timestamps : Optional[torch.Tensor] [E]
        
        Returns
        -------
        logits : torch.Tensor [E]
        """
        # 임베딩 생성
        src_emb = self.get_embeddings(src_nodes, src_features, timestamps)
        dst_emb = self.get_embeddings(dst_nodes, dst_features, timestamps)
        
        # 내적으로 예측
        logits = (src_emb * dst_emb).sum(dim=-1)
        
        return logits
