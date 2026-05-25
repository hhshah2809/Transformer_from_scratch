import torch
import torch.nn.functional as F
from ml.config.training_config import TEMPERATURE, TOP_K, GENERATION_MAX_LENGTH


def top_k_logits(logits, k):
    if k == 0:
        return logits
    v, ix = torch.topk(logits, k)
    minv = v[:, -1].unsqueeze(1)
    return torch.where(logits < minv, torch.full_like(logits, -1e10), logits)


@torch.no_grad()
def generate(model, input_ids, max_length=GENERATION_MAX_LENGTH, temperature=TEMPERATURE, top_k=TOP_K, device="cpu"):
    model.eval()
    model.to(device)

    input_ids = input_ids.to(device)

    generated = input_ids

    for _ in range(max_length):
        logits = model(generated)
        # take last token logits
        next_logits = logits[:, -1, :]

        # apply temperature
        next_logits = next_logits / max(1e-8, temperature)

        # top-k
        next_logits = top_k_logits(next_logits, top_k)

        probs = F.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

    return generated
