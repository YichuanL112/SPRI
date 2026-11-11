from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5ForConditionalGeneration
from ..indexing.index import PAD, EOS
from .shc import StructuredHistoryConstruction
from .guidance import soft_prefix_guidance
from .decoding import beam_search


class SPRIRecommender(nn.Module):
    def __init__(self, index, config):
        super().__init__()
        self.index, self.config = index, config
        t5 = T5Config(vocab_size=index.vocab_size, d_model=config["dim"], d_ff=config["ff_dim"],
                      num_heads=config["heads"], d_kv=config["head_dim"],
                      num_layers=config["encoder_layers"], num_decoder_layers=config["decoder_layers"],
                      dropout_rate=config["dropout"], pad_token_id=PAD,
                      eos_token_id=EOS, decoder_start_token_id=PAD)
        self.t5 = T5ForConditionalGeneration(t5)
        self.shc = StructuredHistoryConstruction(config["dim"], config["shc"])

    def encode(self, history, item_mask):
        inputs, mask = self.shc(history, item_mask, self.t5.shared)
        hidden = self.t5.encoder(inputs_embeds=inputs, attention_mask=mask).last_hidden_state
        return SimpleNamespace(last_hidden_state=hidden, attention_mask=mask)

    def calculate_loss(self, batch, epoch):
        inputs, mask = self.shc(batch["history"], batch["item_mask"], self.t5.shared)
        result = self.t5(inputs_embeds=inputs, attention_mask=mask, labels=batch["labels"], use_cache=False)
        logits = soft_prefix_guidance(result.logits, batch["labels"], self.index.support, epoch,
                                      self.config["guidance"])
        # Full vocabulary CE; hard level-wise masks belong exclusively to decoding.
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))

    def generate(self, batch, width):
        return beam_search(self, batch["history"], batch["item_mask"], width)

