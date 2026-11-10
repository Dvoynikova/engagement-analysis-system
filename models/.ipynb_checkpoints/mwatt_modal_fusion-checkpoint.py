import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)      
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)       
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x
        
class ModalityWeightedAttention(nn.Module):

    def __init__(self,d_qk,seq_len_q=None,seq_len_k=None,dropout=0.1,learnable_M=False,fixed_M=None,init_type="uniform",
                 M_mode="mul",      # "mul", "log_add, "none"
                 M_temperature=1.0  # <<< ДОБАВЛЕНО
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.learnable_M = learnable_M
        self.init_type = init_type
        self.M_mode = M_mode
        self.M_temperature = M_temperature  # <<< ДОБАВЛЕНО

        if fixed_M is not None:
            self.M_fixed = fixed_M
            self.M_param = None
            self.learnable_M = False

        elif learnable_M:
            if seq_len_q is None or seq_len_k is None:
                raise ValueError("Need seq_len for learnable M")
            init_M = self._init_M(seq_len_q, seq_len_k, init_type)

            self.M_param = nn.Parameter(
                torch.logit(init_M.clamp(1e-4, 1 - 1e-4))
            )
            self.M_fixed = None
        else:
            self.M_param = None
            self.M_fixed = None
    def _init_M(self, seq_len_q, seq_len_k, init_type):
        if init_type == "uniform":
            # Рандом в диапазоне 0.8 - 1.0 вместо постоянных 0.9
            M = 0.8 + 0.2 * torch.rand(seq_len_q, seq_len_k)
            
        elif init_type == "diagonal":
            # диагональная матрица с разбросом ±0.1
            M = torch.eye(seq_len_q, seq_len_k) * (0.9 + 0.2 * (torch.rand(seq_len_q, seq_len_k)-0.5))
            
        elif init_type == "local":
            M = torch.zeros(seq_len_q, seq_len_k)
            window = 5
            for i in range(seq_len_q):
                for j in range(seq_len_k):
                    if abs(i - j) <= window:
                        # добавляем шум ±0.05 вокруг 0.9
                        M[i, j] = 0.9 + 0.1 * (torch.rand(1).item() - 0.5)
                        
        elif init_type == "random":
            # случайно от 0 до 1
            M = torch.rand(seq_len_q, seq_len_k)
            
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        return M

    def get_M(self):
        if self.learnable_M:
            #return torch.sigmoid(self.M_param).detach()
            return torch.sigmoid(self.M_temperature * self.M_param).detach()  # <<< ИЗМЕНЕНО
        if self.M_fixed is not None:
            return self.M_fixed.detach()
        return None

    def _apply_M(self, scores, M):
        if self.M_mode == "none":
            return scores
        if self.M_mode == "mul":
            return scores * M
        if self.M_mode == "log_add":
            return scores + torch.log(M + 1e-6)
        raise ValueError(f"Unknown M_mode: {self.M_mode}")

    def forward(self, Q, K, V, M=None, mask=None):
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        
        if M is not None:
            M = torch.clamp(M, 0, 1)
            scores = self._apply_M(scores, M)
        elif self.learnable_M:
            #M = torch.sigmoid(self.M_param)
            M = torch.sigmoid(self.M_temperature * self.M_param)  # <<< ИЗМЕНЕНО
            scores = self._apply_M(scores, M.unsqueeze(0))
        elif self.M_fixed is not None:
            scores = self._apply_M(scores, self.M_fixed.unsqueeze(0))
    
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, scores.size(1), -1)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
    
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.bmm(attention, V)
        return output, attention


class MultiHeadMWAtt(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        T,
        dropout=0.1,
        learnable_M=True,
        init_type="uniform",
        M_mode="mul"
    ):
        super().__init__()

        assert d_model % nhead == 0

        self.nhead = nhead
        self.head_dim = d_model // nhead

        # стандартные Transformer проекции
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        # MWAtt на уровне одной головы
        self.mwatt = ModalityWeightedAttention(
            d_qk=self.head_dim,
            seq_len_q=T,
            seq_len_k=T,
            dropout=dropout,
            learnable_M=learnable_M,
            init_type=init_type,
            M_mode=M_mode
        )

    def _split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.nhead, self.head_dim)
        return x.transpose(1, 2)  # B, H, T, D

    def _merge_heads(self, x):
        B, H, T, D = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * D)

    def forward(self, Q, K, V, mask=None):

        B, T, _ = Q.shape

        Q = self._split_heads(self.q_proj(Q))
        K = self._split_heads(self.k_proj(K))
        V = self._split_heads(self.v_proj(V))

        outputs = []
        attentions = []

        for h in range(self.nhead):

            out, attn = self.mwatt(
                Q[:, h],
                K[:, h],
                V[:, h],
                mask=mask
            )

            outputs.append(out)
            attentions.append(attn)

        outputs = torch.stack(outputs, dim=1)
        x = self._merge_heads(outputs)

        x = self.out_proj(x)

        return x, attentions

    def get_M(self):
        return self.mwatt.get_M()

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.5):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        ff_dim = dim_feedforward or d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x
#Трансформер блок с MWATT
class MWAttTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        T,
        dropout=0.3,
        learnable_M=True,
        init_type="uniform",
        M_mode="mul"
    ):
        super().__init__()

        self.attn = MultiHeadMWAtt(
            d_model=d_model,
            nhead=nhead,
            T=T,
            dropout=dropout,
            learnable_M=learnable_M,
            init_type=init_type,
            M_mode=M_mode
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class MWAttCrossModalTransformer(nn.Module):

    def __init__(
        self,
        video_dim,
        audio_dim,
        d_model=256,
        nhead=8,
        num_layers=2,
        n_classes=3,
        T=125,
        dropout=0.3,
        learnable_M=True,
        init_type="uniform",
        M_mode="mul"
    ):
        super().__init__()

        self.T = T

        # --- Проекции ---
        self.video_proj = nn.Linear(video_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        # --- Позиционное кодирование ---
        self.video_pos = PositionalEncoding(d_model, max_len=T)
        self.audio_pos = PositionalEncoding(d_model, max_len=T)

        # --- Transformer блоки ---
        self.video_layers = nn.ModuleList([
            MWAttTransformerBlock(
                d_model,
                nhead,
                T,
                dropout,
                learnable_M,
                init_type,
                M_mode
            )
            for _ in range(num_layers)
        ])

        self.audio_layers = nn.ModuleList([
            MWAttTransformerBlock(
                d_model,
                nhead,
                T,
                dropout,
                learnable_M,
                init_type,
                M_mode
            )
            for _ in range(num_layers)
        ])

        # --- MWAtt после объединения ---
        fusion_dim = d_model * 2

        self.fusion_mwatt = ModalityWeightedAttention(
            d_qk=fusion_dim,
            seq_len_q=T,
            seq_len_k=T,
            dropout=dropout,
            learnable_M=learnable_M,
            init_type=init_type,
            M_mode=M_mode
        )

        # --- Attention pooling ---
        self.attn_pool = nn.Linear(fusion_dim, 1)

        # --- Классификатор ---
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def get_all_M(self):

        M_dict = {
            "video_layers": [],
            "audio_layers": [],
            "fusion": None
        }
        # video transformer
        for layer in self.video_layers:
            if hasattr(layer, "attn"):
                M = layer.attn.get_M()
                if M is not None:
                    M_dict["video_layers"].append(M.detach().cpu())
        # audio transformer
        for layer in self.audio_layers:
            if hasattr(layer, "attn"):
                M = layer.attn.get_M()
                if M is not None:
                    M_dict["audio_layers"].append(M.detach().cpu())
        # fusion MWAtt
        M = self.fusion_mwatt.get_M()
        if M is not None:
            M_dict["fusion"] = M.detach().cpu()

        return M_dict

    def forward(self, video_x, audio_x, mask):

        padding_mask = (mask == 0)

        # --- Проекции ---
        video = self.video_proj(video_x)
        audio = self.audio_proj(audio_x)

        # --- Positional encoding ---
        video = self.video_pos(video)
        audio = self.audio_pos(audio)

        # --- Transformer блоки ---
        for layer in self.video_layers:
            video = layer(video, mask)

        for layer in self.audio_layers:
            audio = layer(audio, mask)

        # --- Конкатенация ---
        x = torch.cat([video, audio], dim=-1)

        # --- MWAtt fusion ---
        x, _ = self.fusion_mwatt(x, x, x, mask=mask)

        # --- Attention pooling ---
        attn_scores = self.attn_pool(x).squeeze(-1)
        attn_scores = attn_scores.masked_fill(padding_mask, -1e9)

        weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        x = (x * weights).sum(dim=1)

        # --- Классификатор ---
        logits = self.head(x)

        return logits