import torch
import torch.nn as nn
import torch.nn.functional as F


class RhythmMamba(nn.Module):
    """
    Small selective-state rhythm model for the demo.

    Input:  [batch, seq_len, 14] = 12 behavior probabilities + sin/cos time.
    Output: [batch, 12] prior distribution logits for the next time step.
    """

    def __init__(self, input_dim=14, hidden_dim=64, num_classes=12, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.in_proj = nn.Linear(input_dim, hidden_dim * 2)
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_log = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(u)
        gate = torch.sigmoid(gate)

        state = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        a = -torch.exp(self.a_log).view(1, -1)

        for t in range(x.size(1)):
            u_t = u[:, t]
            dt = torch.sigmoid(self.dt_proj(u_t))
            b_t = torch.tanh(self.b_proj(u_t))
            c_t = torch.sigmoid(self.c_proj(u_t))
            state = torch.exp(a * dt) * state + b_t * u_t
            state = (c_t * state) + ((1.0 - c_t) * u_t)

        out = self.norm(state * gate[:, -1])
        out = self.dropout(out)
        return self.head(out)

    @torch.no_grad()
    def predict_prior(self, x):
        return torch.softmax(self.forward(x), dim=-1)
