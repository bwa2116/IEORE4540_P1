class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, num_attention_heads, dropout, bias=True):
        super().__init__()
        # x --> (batch_size, hidden_size, num_patches)
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(hidden_size*num_attention_heads, hidden_size*num_attention_heads)
        # number of random features
        self.m = int(12)
        self.w = nn.Parameter(torch.randn(self.m, attention_head_size), requires_grad = False)

    def prm_exp(self, x):
        # ==== positive random features for gaussian kernels ====
        # x = (batch_size, num_patches, hidden_size)
        # w = (m, hidden_size)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x*x).sum(dim = -1, keepdim = True)).repeat(1, 1, self.m)/2
        wtx = torch.einsum('bti,mi->btm', x, self.w)
        return torch.exp(wtx - xd)/math.sqrt(self.m)

    def forward(self, x):
        kp, qp = self.prm_exp(self.key(x)), self.prm_exp(self.query(x))
        D =  torch.einsum('bti,bi->bt', qp, kp.sum(dim = 1)).unsqueeze(dim = 2)
        kptv = torch.einsum('bin,bim->bnm', self.value(x), kp)
        attention_probs = kptv
        attention_output = torch.einsum('bti,bni->btn', qp, kptv)/D.repeat(1, 1, self.attention_head_size) #(B, T, hidden_size)/Diag
        return (attention_output, attention_probs)
