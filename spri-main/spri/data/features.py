import numpy as np


def encode_fields(split, config, device):
    fields = [[row[key] for row in split["metadata"]] for key in ("category", "title", "fine")]
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(config["model"], revision=config["revision"], device=str(device))
    return [encoder.encode(texts, batch_size=config["batch_size"], normalize_embeddings=config["normalize"],
                           convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
            for texts in fields]

