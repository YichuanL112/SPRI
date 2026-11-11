import torch
from torch.nn import functional as F
from ..artifacts import save_checkpoint, load_checkpoint, rng_state, restore_rng
from ..data.splits import samples
from .din import DeepInterestNetwork


def train_din(split, config, history_length, device, directory, provenance, resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(samples(split, "train", history_length))
    if not rows:
        raise ValueError("No DIN training examples")
    n = len(split["item_ids"])
    histories = torch.full((len(rows), history_length), n, dtype=torch.long)
    targets = torch.tensor([r["target"] for r in rows])
    for i, row in enumerate(rows):
        histories[i, -len(row["history"]):] = torch.tensor(row["history"])
    model = DeepInterestNetwork(n, config["dim"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    first = 0
    last_path = directory / "last.pt"
    if resume:
        state = load_checkpoint(last_path)
        if state["provenance"] != provenance:
            raise ValueError("DIN resume configuration or input mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first = state["epoch"] + 1
        restore_rng(state["rng"])
    for epoch in range(first, config["epochs"]):
        model.train()
        total = 0.0
        order = torch.randperm(len(rows))
        for ids in order.split(config["batch_size"]):
            history, target = histories[ids].to(device), targets[ids].to(device)
            # Sample uniformly from all items except the current positive.
            negative = torch.randint(n - 1, (len(ids), config["negatives"]), device=device)
            negative = negative + (negative >= target[:, None]).long()
            candidates = torch.cat([target[:, None], negative], 1)
            logits = model(history, candidates)
            # Correct the sampled negatives for expected sampling multiplicity.
            correction = torch.log(torch.tensor(config["negatives"] / (n - 1), device=device))
            logits = torch.cat([logits[:, :1], logits[:, 1:] - correction], 1)
            loss = F.cross_entropy(logits, torch.zeros(len(ids), dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        print(f"DIN epoch={epoch + 1} loss={total / len(rows):.6f}", flush=True)
        save_checkpoint(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "epoch": epoch, "rng": rng_state(), "provenance": provenance})
    return model.embedding.weight[:n].detach().cpu()

