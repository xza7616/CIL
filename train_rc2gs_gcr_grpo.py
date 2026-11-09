"""Train RC2GS with component-wise Causal GRPO from scratch.

The original SAR implementation is untouched.  The policy uses four compact,
independently ablatable reward components.  Each component is evaluated on a
factual route and on four one-scale counterfactual interventions, so every
scale action receives a reliable causal advantage without a critic network.
"""

import argparse
import json
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from toolbox import get_dataset, get_logger, runningScore, save_ckpt
from toolbox.models.text4.DINO_RC2GS_GCR_GRPO import EncoderDecoder
from train_rc2gs_sar import (
    depth_integrity_loss,
    save_named_checkpoint,
    seed_everything,
    segmentation_objective,
)

SUIM_CLASS_NAMES = ("BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR")
REWARD_NAMES = ("overlap", "macc", "tail", "boundary")


@torch.no_grad()
def four_component_reward(logits, target, num_classes, soft_iou_blend=0.25):
    """Return [B,4]: overlap, macro accuracy, tail IoU and boundary F1.

    Hard and soft mIoU are fused inside one overlap component, rather than
    counted as two separate rewards.  This leaves exactly four reward families
    for clean ablation while keeping route comparisons smooth.
    """
    probability = logits.softmax(dim=1).float()
    prediction = probability.argmax(dim=1)
    blend = min(max(float(soft_iou_blend), 0.0), 1.0)
    output = []

    for pred, truth, prob in zip(prediction, target, probability):
        hard_ious = []
        recalls = []
        soft_ious = []
        for class_id in range(num_classes):
            pred_class = pred == class_id
            truth_class = truth == class_id
            truth_count = truth_class.sum()
            union = (pred_class | truth_class).sum()

            if union.item() > 0:
                hard_ious.append(
                    (pred_class & truth_class).sum().float() / union.float()
                )
            if truth_count.item() > 0:
                recalls.append(
                    (pred_class & truth_class).sum().float()
                    / truth_count.float()
                )
                truth_float = truth_class.float()
                class_probability = prob[class_id]
                intersection = (class_probability * truth_float).sum()
                soft_union = (
                    class_probability.sum() + truth_float.sum() - intersection
                )
                soft_ious.append((intersection + 1.0) / (soft_union + 1.0))

        zero = prob.new_zeros(())
        hard_miou = torch.stack(hard_ious).mean() if hard_ious else zero
        soft_miou = torch.stack(soft_ious).mean() if soft_ious else zero
        overlap = (1.0 - blend) * hard_miou + blend * soft_miou
        macc = torch.stack(recalls).mean() if recalls else zero

        if hard_ious:
            sorted_iou = torch.sort(torch.stack(hard_ious))[0]
            tail_count = max(1, (len(hard_ious) + 2) // 3)
            tail_iou = sorted_iou[:tail_count].mean()
        else:
            tail_iou = zero

        pred_edge = torch.zeros_like(pred, dtype=torch.bool)
        truth_edge = torch.zeros_like(truth, dtype=torch.bool)
        pred_edge[:, 1:] |= pred[:, 1:] != pred[:, :-1]
        pred_edge[1:, :] |= pred[1:, :] != pred[:-1, :]
        truth_edge[:, 1:] |= truth[:, 1:] != truth[:, :-1]
        truth_edge[1:, :] |= truth[1:, :] != truth[:-1, :]
        pred_dilated = F.max_pool2d(
            pred_edge.float()[None, None], 3, 1, 1
        ).bool()[0, 0]
        truth_dilated = F.max_pool2d(
            truth_edge.float()[None, None], 3, 1, 1
        ).bool()[0, 0]
        precision = (pred_edge & truth_dilated).sum().float() / (
            pred_edge.sum().float() + 1e-6
        )
        recall = (truth_edge & pred_dilated).sum().float() / (
            truth_edge.sum().float() + 1e-6
        )
        boundary_f1 = 2.0 * precision * recall / (
            precision + recall + 1e-6
        )
        output.append(torch.stack((overlap, macc, tail_iou, boundary_f1)))

    return torch.stack(output, dim=0)


def reward_weights(reference, cfg):
    weights = reference.new_tensor(
        (
            cfg.get("reward_overlap_weight", 1.00),
            cfg.get("reward_macc_weight", 0.25),
            cfg.get("reward_tail_weight", 0.10),
            cfg.get("reward_boundary_weight", 0.10),
        )
    ).clamp_min(0.0)
    if weights.sum().item() <= 0.0:
        raise ValueError("At least one of the four reward weights must be positive")
    return weights


@torch.no_grad()
def reward_group(logits, target, cfg):
    """Evaluate four reward components and their normalized scalarization."""
    components = torch.stack(
        [
            four_component_reward(
                candidate,
                target,
                cfg["n_classes"],
                cfg.get("reward_overlap_soft_blend", 0.25),
            )
            for candidate in logits
        ],
        dim=0,
    )
    weights = reward_weights(components, cfg)
    normalized_weights = weights / weights.sum()
    total = (
        components * normalized_weights.reshape(1, 1, -1)
    ).sum(dim=-1)
    return total, components, weights


@torch.no_grad()
def validate_detailed(model, loader, device, num_classes):
    model.eval()
    metric = runningScore(num_classes, ignore_index=None)
    losses = []
    for sample in loader:
        image = sample["image"].to(device, non_blocking=True)
        depth = sample["depth"].to(device, non_blocking=True)
        target = sample["label"].to(device, non_blocking=True)
        logits = model(image, depth)
        losses.append(F.cross_entropy(logits, target).item())
        metric.update(target.cpu().numpy(), logits.argmax(1).cpu().numpy())
    scores, _, class_accuracy = metric.get_scores()
    mean_loss = sum(losses) / max(len(losses), 1)
    per_class = [float(class_accuracy[index]) for index in range(num_classes)]
    return mean_loss, scores["mIou: "], scores["mAcc: "], per_class


def _reliable_group_advantage(values, minimum_spread, maximum_advantage):
    """Normalize over sampled routes and suppress near-zero-spread noise."""
    centered = values - values.mean(dim=0, keepdim=True)
    spread = values.std(dim=0, keepdim=True, unbiased=False)
    reliability = spread / (spread + max(float(minimum_spread), 1e-6))
    advantage = (centered / (spread + 1e-6) * reliability).clamp(
        -float(maximum_advantage), float(maximum_advantage)
    )
    return advantage, spread, reliability


def component_causal_grpo_loss(
    scale_log_probs,
    rewards,
    causal_components,
    weights,
    causal_strength,
    cfg,
):
    """Use a scale-specific global-plus-causal advantage for every action.

    Causal credit is first normalized independently for all four reward
    components.  Component reliabilities form a causal gate, so an uncertain
    intervention cannot overpower the stable group-relative advantage.
    """
    maximum_advantage = cfg.get("grpo_maximum_advantage", 2.0)
    global_advantage, global_spread, global_reliability = (
        _reliable_group_advantage(
            rewards,
            cfg.get("grpo_minimum_reward_spread", 0.01),
            maximum_advantage,
        )
    )

    zero = scale_log_probs.new_zeros(())
    normalized_weights = weights / weights.sum()
    if float(causal_strength) > 0.0:
        component_advantage, causal_spread, component_reliability = (
            _reliable_group_advantage(
                causal_components,
                cfg.get("causal_minimum_reward_spread", 0.006),
                maximum_advantage,
            )
        )
        component_weight = normalized_weights.reshape(1, 1, 1, -1)
        causal_advantage = (
            component_advantage * component_weight
        ).sum(dim=-1)
        causal_reliability = (
            component_reliability * component_weight
        ).sum(dim=-1)
        causal_gate = (
            global_reliability.unsqueeze(-1) * causal_reliability
        ).clamp(0.0, 1.0).sqrt()
    else:
        causal_advantage = torch.zeros_like(scale_log_probs)
        causal_spread = causal_components.new_zeros(
            (1,) + causal_components.shape[1:]
        )
        component_reliability = causal_spread
        causal_reliability = causal_components.new_zeros(
            (1,) + causal_components.shape[1:-1]
        )
        causal_gate = causal_reliability

    # At causal_strength=0, this gives the same per-action gradient as the
    # original joint-log-probability GRPO.  When active, every scale receives
    # its own counterfactual correction rather than a shared route reward.
    combined_advantage = (
        global_advantage.unsqueeze(-1)
        + float(causal_strength)
        * causal_gate.detach()
        * causal_advantage
    )
    old_scale_log_probs = scale_log_probs.detach()
    ratio = (scale_log_probs - old_scale_log_probs).exp()
    clip_epsilon = float(cfg.get("grpo_clip_epsilon", 0.2))
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped = ratio * combined_advantage.detach()
    clipped = clipped_ratio * combined_advantage.detach()
    loss = -torch.minimum(unclipped, clipped).sum(dim=-1).mean()

    global_surrogate = ratio * global_advantage.unsqueeze(-1).detach()
    global_loss = -global_surrogate.sum(dim=-1).mean()
    if float(causal_strength) > 0.0:
        causal_surrogate = (
            ratio * causal_gate.detach() * causal_advantage.detach()
        )
        causal_loss = -causal_surrogate.sum(dim=-1).mean()
    else:
        causal_loss = zero

    return {
        "loss": loss,
        "global_loss": global_loss,
        "causal_loss": causal_loss,
        "global_spread": global_spread.mean(),
        "causal_spread": causal_spread.mean(),
        "global_reliability": global_reliability.mean(),
        "causal_reliability": causal_reliability.mean(),
        "causal_gate": causal_gate.mean(),
        "component_reliability": component_reliability.mean(
            dim=tuple(range(component_reliability.ndim - 1))
        ),
    }


@torch.no_grad()
def counterfactual_component_credit(group, target, cfg):
    """Return causal effects for four scales and four reward components."""
    if group["credit_logits"] is None or group["counterfactual_logits"] is None:
        raise ValueError("Counterfactual logits were not requested by the model")

    native_height, native_width = group["credit_logits"].shape[-2:]
    reward_scale = float(cfg.get("credit_reward_scale", 0.25))
    if reward_scale > 0.0:
        height = max(native_height, int(round(target.shape[-2] * reward_scale)))
        width = max(native_width, int(round(target.shape[-1] * reward_scale)))
    else:
        height, width = native_height, native_width

    def resize_logits(logits):
        if logits.shape[-2:] == (height, width):
            return logits
        leading = logits.shape[:-3]
        channels = logits.shape[-3]
        resized = F.interpolate(
            logits.reshape(-1, channels, native_height, native_width),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(*leading, channels, height, width)

    credit_target = F.interpolate(
        target.unsqueeze(1).float(),
        size=(height, width),
        mode="nearest",
    ).squeeze(1).long()
    factual_reward, factual_components, weights = reward_group(
        resize_logits(group["credit_logits"]), credit_target, cfg
    )

    counterfactual_rewards = []
    counterfactual_components = []
    for scale_logits in group["counterfactual_logits"]:
        scale_reward, scale_components, _ = reward_group(
            resize_logits(scale_logits), credit_target, cfg
        )
        counterfactual_rewards.append(scale_reward)
        counterfactual_components.append(scale_components)

    # [scale,group,batch,(component)] -> [group,batch,scale,(component)]
    scalar_credit = (
        factual_reward.unsqueeze(0)
        - torch.stack(counterfactual_rewards, dim=0)
    ).permute(1, 2, 0).contiguous()
    component_credit = (
        factual_components.unsqueeze(0)
        - torch.stack(counterfactual_components, dim=0)
    ).permute(1, 2, 0, 3).contiguous()
    return {
        "total": scalar_credit,
        "components": component_credit,
        "weights": weights,
    }


def causal_strength_at_epoch(epoch, cfg):
    """Delay causal optimization and smoothly introduce its correction."""
    start_epoch = int(cfg.get("causal_start_epoch", 8))
    ramp_epochs = max(int(cfg.get("causal_ramp_epochs", 12)), 1)
    maximum = float(cfg.get("causal_weight", 0.15))
    current_epoch = int(epoch) + 1
    if current_epoch <= start_epoch:
        return 0.0
    progress = min(float(current_epoch - start_epoch) / ramp_epochs, 1.0)
    return maximum * progress


def build_model(cfg):
    return EncoderDecoder(
        num_classes=cfg.get("n_classes", 8),
        freeze_backbone=cfg.get("freeze_backbone", True),
        stride=cfg.get("stride", 16),
        feature_dim=cfg.get("feature_dim", 256),
        gaussian_kernel_size=cfg.get("gaussian_kernel_size", 3),
        gaussian_query_dim=cfg.get("gaussian_query_dim", 32),
        local_router_hidden=cfg.get("local_router_hidden", 64),
        local_blend_max=cfg.get("local_blend_max", 0.35),
        local_balance_weight=cfg.get("local_balance_weight", 0.01),
        dino_repo=cfg.get("dino_repo"),
        dino_weights=cfg.get("dino_weights"),
    )


def train(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if cfg.get("n_classes") != 8:
        raise ValueError("SUIM-D must be configured with n_classes=8")
    if cfg.get("model_name") != "DINO_RC2GS_GCR_GRPO":
        raise ValueError("This trainer requires model_name=DINO_RC2GS_GCR_GRPO")

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(
        args.output, time.strftime("%Y-%m-%d-%H-%M-%S") + "-rc2gs-causal-grpo"
    )
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, run_dir)
    logger = get_logger(run_dir)
    logger.info("Run directory: %s", run_dir)
    logger.info("Training from scratch; original SAR files/checkpoints are untouched")
    logger.info(
        "Causal-GRPO | rewards=[overlap %.2f, mAcc %.2f, tail %.2f, "
        "boundary %.2f] | softIoUBlend=%.2f | causal start=%d ramp=%d "
        "maxWeight=%.3f | creditScale=%.2f",
        cfg.get("reward_overlap_weight", 1.00),
        cfg.get("reward_macc_weight", 0.25),
        cfg.get("reward_tail_weight", 0.10),
        cfg.get("reward_boundary_weight", 0.10),
        cfg.get("reward_overlap_soft_blend", 0.25),
        cfg.get("causal_start_epoch", 8),
        cfg.get("causal_ramp_epochs", 12),
        cfg.get("causal_weight", 0.15),
        cfg.get("credit_reward_scale", 0.25),
    )

    train_set, val_set = get_dataset(cfg)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["ims_per_gpu"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = build_model(cfg).to(device)
    core_model = model.module if hasattr(model, "module") else model

    class_weight = None
    if hasattr(train_set, "class_weight"):
        candidate = torch.as_tensor(
            train_set.class_weight, dtype=torch.float32, device=device
        ).reshape(-1)
        if candidate.numel() == cfg["n_classes"]:
            class_weight = candidate
        else:
            logger.warning(
                "Ignoring incompatible class_weight with %d entries",
                candidate.numel(),
            )

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["lr_start"],
        weight_decay=cfg["weight_decay"],
    )
    warmup_epochs = cfg.get("warmup_epochs", 3)
    minimum_ratio = cfg.get("minimum_lr_ratio", 0.05)

    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = float(epoch - warmup_epochs) / max(
            cfg["epochs"] - warmup_epochs - 1, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_schedule
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_miou = -1.0
    best_miou_epoch = 0
    macc_threshold = float(cfg.get("macc_summary_threshold", 0.895))
    high_macc_epochs = []
    best_qualified_miou = -1.0
    best_qualified_epoch = 0
    best_qualified_macc = 0.0

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        model.train()
        metric_keys = (
            "loss",
            "main",
            "route",
            "reward",
            "causal_credit",
            "global_loss",
            "causal_loss",
            "global_spread",
            "causal_spread",
            "global_reliability",
            "causal_reliability",
            "causal_gate",
            "local_entropy",
            "local_blend",
        )
        component_keys = tuple(
            "reward_" + name for name in REWARD_NAMES
        ) + tuple("causal_" + name for name in REWARD_NAMES)
        totals = {key: 0.0 for key in metric_keys + component_keys}

        policy_scale = min(
            1.0,
            float(epoch + 1) / max(cfg.get("grpo_warmup_epochs", 6), 1),
        )
        entropy_scale = max(
            0.0, 1.0 - float(epoch) / max(cfg["epochs"] - 1, 1)
        )
        causal_strength = causal_strength_at_epoch(epoch, cfg)

        for step, sample in enumerate(train_loader):
            image = sample["image"].to(device, non_blocking=True)
            depth = sample["depth"].to(device, non_blocking=True)
            target = sample["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                group = model.forward_group(
                    image,
                    depth,
                    group_size=cfg.get("grpo_group_size", 4),
                    temperature=cfg.get("grpo_temperature", 1.0),
                    compute_counterfactual=causal_strength > 0.0,
                )
                logits = group["expected_logits"]
                main_loss = segmentation_objective(
                    logits, target, class_weight, cfg
                )
                route_loss = torch.stack(
                    [
                        segmentation_objective(
                            candidate, target, class_weight, cfg
                        )
                        for candidate in group["logits"]
                    ]
                ).mean()
                integrity = depth_integrity_loss(logits, target, depth)
                segmentation_loss = (
                    main_loss
                    + cfg.get("group_supervision_weight", 0.20) * route_loss
                    + cfg.get("depth_integrity_weight", 0.04) * integrity
                )

                rewards, reward_components, weights = reward_group(
                    group["logits"], target, cfg
                )
                if causal_strength > 0.0:
                    causal = counterfactual_component_credit(
                        group, target, cfg
                    )
                    causal_credit = causal["total"]
                    causal_components = causal["components"]
                else:
                    causal_credit = torch.zeros_like(group["scale_log_probs"])
                    causal_components = reward_components.new_zeros(
                        reward_components.shape[0],
                        reward_components.shape[1],
                        group["scale_log_probs"].shape[-1],
                        reward_components.shape[-1],
                    )

                policy = component_causal_grpo_loss(
                    group["scale_log_probs"],
                    rewards,
                    causal_components,
                    weights,
                    causal_strength,
                    cfg,
                )
                loss = (
                    segmentation_loss
                    + cfg.get("gaussian_aux_weight", 0.05)
                    * group["auxiliary_loss"]
                    + policy_scale
                    * cfg.get("grpo_weight", 0.05)
                    * policy["loss"]
                    - entropy_scale
                    * cfg.get("entropy_weight", 0.001)
                    * group["entropy"]
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable, cfg.get("max_grad_norm", 1.0)
            )
            scaler.step(optimizer)
            scaler.update()

            local_entropy, local_blend = core_model.local_route_statistics()
            values = {
                "loss": loss.item(),
                "main": main_loss.item(),
                "route": route_loss.item(),
                "reward": rewards.mean().item(),
                "causal_credit": causal_credit.abs().mean().item(),
                "global_loss": policy["global_loss"].item(),
                "causal_loss": policy["causal_loss"].item(),
                "global_spread": policy["global_spread"].item(),
                "causal_spread": policy["causal_spread"].item(),
                "global_reliability": policy["global_reliability"].item(),
                "causal_reliability": policy["causal_reliability"].item(),
                "causal_gate": policy["causal_gate"].item(),
                "local_entropy": local_entropy.item(),
                "local_blend": local_blend.item(),
            }
            for index, name in enumerate(REWARD_NAMES):
                values["reward_" + name] = (
                    reward_components[..., index].mean().item()
                )
                values["causal_" + name] = (
                    causal_components[..., index].abs().mean().item()
                )
            for key, value in values.items():
                totals[key] += value

            if (step + 1) % args.log_interval == 0:
                count = step + 1
                logger.info(
                    "Train | epoch [%d/%d] step [%d/%d] loss=%.4f "
                    "main=%.4f route=%.4f reward=%.4f causal=%.5f "
                    "gSpread=%.5f cSpread=%.5f gRel=%.3f cRel=%.3f "
                    "cGate=%.3f causalScale=%.3f policyScale=%.2f lr=%.6g",
                    epoch + 1,
                    cfg["epochs"],
                    count,
                    len(train_loader),
                    totals["loss"] / count,
                    totals["main"] / count,
                    totals["route"] / count,
                    totals["reward"] / count,
                    totals["causal_credit"] / count,
                    totals["global_spread"] / count,
                    totals["causal_spread"] / count,
                    totals["global_reliability"] / count,
                    totals["causal_reliability"] / count,
                    totals["causal_gate"] / count,
                    causal_strength,
                    policy_scale,
                    optimizer.param_groups[0]["lr"],
                )

        scheduler.step()
        val_loss, miou, macc, class_accuracy = validate_detailed(
            model, val_loader, device, cfg["n_classes"]
        )

        is_high_macc = macc > macc_threshold
        if is_high_macc:
            high_macc_epochs.append((epoch + 1, macc, miou))
            save_named_checkpoint(
                run_dir, model, "model_{}.pth".format(epoch + 1)
            )
            if miou > best_qualified_miou:
                best_qualified_miou = miou
                best_qualified_epoch = epoch + 1
                best_qualified_macc = macc
                save_named_checkpoint(
                    run_dir, model, "model_macc895_best.pth"
                )

        save_ckpt(run_dir, model, kind="end")
        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            best_miou_epoch = epoch + 1
            save_ckpt(run_dir, model, kind="best")

        batches = max(len(train_loader), 1)
        logger.info(
            "Epoch | [%d/%d] train_loss=%.5f main=%.5f route=%.5f "
            "reward=%.5f causal=%.5f gLoss=%.5f cLoss=%.5f "
            "gSpread=%.5f cSpread=%.5f gRel=%.3f cRel=%.3f "
            "cGate=%.3f causalScale=%.3f localBlend=%.3f "
            "localEntropy=%.3f val_loss=%.5f mAcc=%.4f mIoU=%.4f "
            "best_mIoU=%.4f@epoch%d time=%.1fs%s",
            epoch + 1,
            cfg["epochs"],
            totals["loss"] / batches,
            totals["main"] / batches,
            totals["route"] / batches,
            totals["reward"] / batches,
            totals["causal_credit"] / batches,
            totals["global_loss"] / batches,
            totals["causal_loss"] / batches,
            totals["global_spread"] / batches,
            totals["causal_spread"] / batches,
            totals["global_reliability"] / batches,
            totals["causal_reliability"] / batches,
            totals["causal_gate"] / batches,
            causal_strength,
            totals["local_blend"] / batches,
            totals["local_entropy"] / batches,
            val_loss,
            macc,
            miou,
            best_miou,
            best_miou_epoch,
            time.time() - epoch_start,
            " | new best saved" if is_best else "",
        )
        logger.info(
            "Reward4 | overlap=%.4f mAcc=%.4f tail=%.4f boundary=%.4f | "
            "Causal4(abs) | overlap=%.5f mAcc=%.5f tail=%.5f boundary=%.5f",
            totals["reward_overlap"] / batches,
            totals["reward_macc"] / batches,
            totals["reward_tail"] / batches,
            totals["reward_boundary"] / batches,
            totals["causal_overlap"] / batches,
            totals["causal_macc"] / batches,
            totals["causal_tail"] / batches,
            totals["causal_boundary"] / batches,
        )
        class_text = " ".join(
            "{}={:.4f}".format(name, accuracy)
            for name, accuracy in zip(SUIM_CLASS_NAMES, class_accuracy)
        )
        logger.info("ClassAcc | epoch%d | %s", epoch + 1, class_text)

    logger.info(
        "Training complete | best_mIoU=%.4f@epoch%d",
        best_miou,
        best_miou_epoch,
    )
    summary = (
        ", ".join(
            "epoch%d(mAcc=%.4f,mIoU=%.4f)" % item
            for item in high_macc_epochs
        )
        if high_macc_epochs
        else "none"
    )
    logger.info(
        "mAcc > %.4f epochs | count=%d | %s",
        macc_threshold,
        len(high_macc_epochs),
        summary,
    )
    if best_qualified_epoch > 0:
        logger.info(
            "Selected constrained checkpoint | model_macc895_best.pth | "
            "epoch%d mAcc=%.4f mIoU=%.4f",
            best_qualified_epoch,
            best_qualified_macc,
            best_qualified_miou,
        )
    else:
        logger.info(
            "Selected constrained checkpoint | none "
            "(no epoch with mAcc > %.4f)",
            macc_threshold,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/SUIM_RC2GS_GCR_GRPO.json"
    )
    parser.add_argument("--output", default="runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-interval", type=int, default=20)
    train(parser.parse_args())
