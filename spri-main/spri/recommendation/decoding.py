"""Fixed-width beam search; unsupported tuples retain their generated rank."""
import torch
from transformers.modeling_outputs import BaseModelOutput


@torch.no_grad()
def beam_search(model, history, item_mask, width):
    encoder = model.encode(history, item_mask)
    states, mask = encoder.last_hidden_state, encoder.attention_mask
    batch = len(history)
    sequences = torch.full((batch, 1, 1), model.t5.config.decoder_start_token_id,
                           device=history.device, dtype=torch.long)
    scores = torch.zeros((batch, 1), device=history.device)
    for level in range(4):
        beams = sequences.shape[1]
        result = model.t5(
            encoder_outputs=BaseModelOutput(last_hidden_state=states.repeat_interleave(beams, 0)),
            attention_mask=mask.repeat_interleave(beams, 0),
            decoder_input_ids=sequences.reshape(batch * beams, -1), use_cache=False)
        start = model.index.offsets[level]
        size = model.index.sizes[level]
        # Softmax after masking is equivalent to normalizing only the current span.
        log_probs = result.logits[:, -1, start:start + size].float().log_softmax(-1)
        candidate_scores = scores[..., None] + log_probs.reshape(batch, beams, size)
        take = min(width, beams * size)
        scores, positions = candidate_scores.flatten(1).topk(take, dim=-1)
        parents, tokens = positions // size, positions % size + start
        sequences = torch.cat([sequences.gather(1, parents[..., None].expand(-1, -1, sequences.shape[-1])),
                               tokens[..., None]], -1)
    # EOS is forced next, contributes log(1)=0, and is omitted from returned identifiers.
    return sequences[:, :, 1:], scores

