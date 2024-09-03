import torch
import torchaudio

def get_param_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    requires_grad: bool = False,
    peak_normalize: bool = False,
    dropout: float = 0.0,
):
    bs, chs, seq_len = x.shape

    x_device = x

    # move audio to model device
    x = x.type_as(next(model.parameters()))

    # if peak_normalize:
    #    x = batch_peak_normalize(x)

    if sample_rate != 48000:
        x = torchaudio.functional.resample(x, sample_rate, 48000)

    seq_len = x.shape[-1]  # update seq_len after resampling
    # if longer than 262144 crop, else repeat pad to 262144
    # if seq_len > 262144:
    #    x = x[:, :, :262144]
    # else:
    #    x = torch.nn.functional.pad(x, (0, 262144 - seq_len), "replicate")

    # peak normalize each batch item
    for batch_idx in range(bs):
        x[batch_idx, ...] /= x[batch_idx, ...].abs().max().clamp(1e-8)

    if not requires_grad:
        with torch.no_grad():
            mid_embeddings, side_embeddings = model(x)
    else:
        mid_embeddings, side_embeddings = model(x)

    # add dropout
    if dropout > 0.0:
        mid_embeddings = torch.nn.functional.dropout(
            mid_embeddings, p=dropout, training=True
        )
        side_embeddings = torch.nn.functional.dropout(
            side_embeddings, p=dropout, training=True
        )

    # check for nan
    if torch.isnan(mid_embeddings).any():
        print("Warning: NaNs found in mid_embeddings")
        mid_embeddings = torch.nan_to_num(mid_embeddings)
    elif torch.isnan(side_embeddings).any():
        print("Warning: NaNs found in side_embeddings")
        side_embeddings = torch.nan_to_num(side_embeddings)

    # l2 normalize
    mid_embeddings = torch.nn.functional.normalize(mid_embeddings, p=2, dim=-1)
    side_embeddings = torch.nn.functional.normalize(side_embeddings, p=2, dim=-1)

    embeddings = {
        "mid": mid_embeddings.type_as(x_device),
        "side": side_embeddings.type_as(x_device),
    }

    return embeddings