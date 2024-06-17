import torch
from gigagan_pytorch.gigagan_pytorch import *


class PseudoTextEncoder(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim_in,
        dim,
        depth,
        seq_len = 4,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        self.encoder = nn.Linear(dim_in, seq_len * dim, bias=False)
        set_requires_grad_(self.encoder, False)
        self.learned_global_token = nn.Parameter(torch.randn(dim))

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads
        )

    @beartype
    def forward(
        self,
        texts: List[Tensor] | None = None,
        text_encodings: Tensor | None = None
    ):
        assert exists(texts) ^ exists(text_encodings)

        if not exists(text_encodings):
            text_encodings = self.encoder(torch.cat(texts, dim=-1)).view(-1, self.seq_len, self.dim)

        mask = (text_encodings != 0.).any(dim = -1)

        mask_with_global = F.pad(mask, (1, 0), value = True)

        batch = text_encodings.shape[0]
        global_tokens = repeat(self.learned_global_token, 'd -> b d', b = batch)

        text_encodings, ps = pack([global_tokens, text_encodings], 'b * d')

        text_encodings = self.transformer(text_encodings, mask = mask_with_global)

        global_tokens, text_encodings = unpack(text_encodings, ps, 'b * d')

        return global_tokens, text_encodings, mask


