import json
import math
import torch
from torch.utils.data import DataLoader
from ..artifacts import save_checkpoint, load_checkpoint, rng_state, restore_rng
from ..evaluation.metrics import evaluate


def train_generator(model, train_data, valid_data, config, evaluation, device, directory, provenance, resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config["eval_batch_size"])
    if len(train_loader) == 0:
        raise ValueError("No generator training examples")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    updates = math.ceil(len(train_loader) / config["accumulation"]) * config["epochs"]
    def schedule(step):
        if step < config["warmup_steps"]:
            return (step + 1) / max(1, config["warmup_steps"])
        return max(0.0, (updates - step) / max(1, updates - config["warmup_steps"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best, stale, first, step = float("-inf"), 0, 0, 0
    if resume:
        state = load_checkpoint(directory / "last.pt")
        if state["provenance"] != provenance:
            raise ValueError("Generator resume configuration or index mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, stale, first, step = state["best"], state["stale"], state["epoch"] + 1, state["step"]
        restore_rng(state["rng"])
    for epoch in range(first, config["epochs"]):
        if stale >= config["patience"]:
            break
        model.train()
        total = count = 0
        optimizer.zero_grad()
        for batch_number, batch in enumerate(train_loader):
            data = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            loss = model.calculate_loss(data, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite generator loss")
            # Weight partial final batches by examples, including partial accumulation groups.
            group_start = batch_number // config["accumulation"] * config["accumulation"]
            group_examples = min(config["accumulation"] * config["batch_size"],
                                 len(train_data) - group_start * config["batch_size"])
            (loss * len(data["labels"]) / group_examples).backward()
            if (batch_number + 1) % config["accumulation"] == 0 or batch_number + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
            total += float(loss.detach()) * len(data["labels"])
            count += len(data["labels"])
        metrics, _ = evaluate(model, valid_loader, device, evaluation, model.index)
        score = metrics[evaluation["selection_metric"]]
        improved = score > best
        best = max(best, score)
        stale = 0 if improved else stale + 1
        record = {"epoch": epoch + 1, "loss": total / count, "validation": metrics}
        print("GENERATOR " + json.dumps(record), flush=True)
        with (directory / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step,
                 "best": best, "stale": stale, "rng": rng_state(), "provenance": provenance,
                 "model_config": model.config, "validation": metrics}
        if improved:
            save_checkpoint(directory / "best.pt", state)
        save_checkpoint(directory / "last.pt", state)
    return load_checkpoint(directory / "best.pt")

