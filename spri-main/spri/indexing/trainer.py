import json
import torch
from ..artifacts import save_checkpoint, load_checkpoint, rng_state, restore_rng
from .tokenizer import SPRIIndexTokenizer
from .losses import alignment, ramp


def train_index(fields, config, device, directory, provenance, resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    fields = [f.to(device) for f in fields]
    model = SPRIIndexTokenizer([f.shape[1] for f in fields], config).to(device)
    state = None
    if resume:
        state = load_checkpoint(directory / "last.pt")
        if state["provenance"] != provenance:
            raise ValueError("Tokenizer resume configuration or input mismatch")
        model.load_state_dict(state["model"])
    else:
        model.initialize_codebooks(fields)
    for stage, epochs in ((1, config["stage1_epochs"]), (2, config["stage2_epochs"])):
        if state is not None and stage < state["stage"]:
            continue
        continuing = state is not None and state["stage"] == stage
        first = state["epoch"] + 1 if continuing else 0
        if stage == 2 and not continuing:
            model.initialize_behavior(fields)
        # Fresh optimizer at Stage 2; all parameters stay in groups through unfreezing.
        model.freeze_shallow(False)
        codebooks, other = [], []
        for name, parameter in model.named_parameters():
            (codebooks if ".embedding.weight" in name else other).append(parameter)
        optimizer = torch.optim.AdamW([{"params": other, "lr": config["lr"]},
                                       {"params": codebooks, "lr": config["codebook_lr"]}],
                                      weight_decay=config["weight_decay"])
        if continuing:
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
        for epoch in range(first, epochs):
            model.train()
            model.freeze_shallow(stage == 2 and epoch < config["unfreeze_epoch"])
            sums, count = {}, 0
            for ids in torch.randperm(len(fields[0]), device=device).split(config["batch_size"]):
                batch = [x[ids] for x in fields]
                out = model(batch, stage)
                loss = out["loss"]
                terms = {"base": loss}
                if stage == 2:
                    a = config["alignment"]
                    terms["alignment"] = a["weight"] * ramp(epoch, a["start"], a["warmup"]) * alignment(out["latents"][2], out["latents"][3], a)
                    s = config["sem2beh"]
                    if s["enabled"]:
                        terms["sem2beh"] = s["weight"] * ramp(epoch, s["start"], s["warmup"]) * model.sem2beh(
                            out["quantized"][2], batch[3], s["detach_source"])
                loss = sum(terms.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite tokenizer loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                for key, value in terms.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach()) * len(ids)
                count += len(ids)
            metrics = {"stage": stage, "epoch": epoch + 1, **{k: v / count for k, v in sums.items()}}
            print("INDEX " + json.dumps(metrics), flush=True)
            with (directory / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "stage": stage, "epoch": epoch, "rng": rng_state(),
                          "input_dims": model.input_dims, "config": config, "provenance": provenance}
            save_checkpoint(directory / "last.pt", checkpoint)
            if epoch == epochs - 1:
                save_checkpoint(directory / f"stage{stage}.pt", checkpoint)
    model.freeze_shallow(False)
    return model

