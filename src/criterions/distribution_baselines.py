"""Embedding-level MMD, CORAL, and direct SW baselines for DAVED.

No GMM or characteristic functions. The frozen teacher defines the coordinate
system; a wrapper-owned s2t projector maps only the student into that system.
All losses use float32 and global batches with differentiable student gathers.
"""

import math

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather as grad_all_gather


def _check_pair(x, y, min_samples=1):
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Expected [N, D] and [M, D] in the SAME coordinate system")
    if min(x.shape[0], y.shape[0]) < min_samples:
        raise ValueError(f"Both sets need at least {min_samples} samples")
    if x.device != y.device:
        raise ValueError("Student and teacher tensors must be on the same device")


def _squared_distances(x, y):
    return (x.square().sum(1, keepdim=True)
            + y.square().sum(1).unsqueeze(0) - 2 * x @ y.T).clamp_min(0)


def mmd_loss(x, y, bandwidth=0.0, scales=(0.5, 1.0, 2.0)):
    """Biased MMD^2 V-statistic with an averaged multi-RBF kernel.

    bandwidth is sigma, not sigma^2. Zero selects a detached median-distance
    bandwidth on the pooled batch. scales multiply sigma. Includes diagonals;
    finite-batch values for independent draws from the same law need not be zero.
    """
    _check_pair(x, y)
    if bandwidth < 0 or not math.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and >= 0")
    if not scales or any(s <= 0 or not math.isfinite(s) for s in scales):
        raise ValueError("RBF scales must be finite and positive")
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.detach().float()
        dxx, dyy, dxy = (_squared_distances(x, x),
                          _squared_distances(y, y), _squared_distances(x, y))
        if bandwidth == 0:
            with torch.no_grad():
                z = torch.cat((x, y))
                d = _squared_distances(z, z)
                upper = d[torch.triu_indices(len(z), len(z), 1, device=z.device).unbind()]
                positive = upper[upper > 0]
                sigma2 = positive.median() if positive.numel() else d.new_tensor(1.0)
                sigma2 = sigma2.clamp_min(1e-12)
        else:
            sigma2 = x.new_tensor(bandwidth ** 2)
        result = x.new_zeros(())
        for scale in scales:
            denom = 2 * sigma2 * scale ** 2
            result = result + ((-dxx / denom).exp().mean()
                               + (-dyy / denom).exp().mean()
                               - 2 * (-dxy / denom).exp().mean())
        return (result / len(scales)).clamp_min(0)


def coral_loss(x, y):
    """Deep CORAL: ||Cov(x)-Cov(y)||_F^2 / (4 D^2); no mean penalty."""
    _check_pair(x, y, min_samples=2)
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.detach().float()
        xc, yc = x - x.mean(0), y - y.mean(0)
        cs = xc.T @ xc / (len(x) - 1)
        ct = yc.T @ yc / (len(y) - 1)
        return (cs - ct).square().mean() / 4


def sliced_wasserstein_loss(x, y, num_slices=256, seed=42):
    """Monte Carlo sliced W2^2 using shared unit directions and exact sorting.

    Equal sample counts are required (paired DAVED caches satisfy this).
    This is sliced Wasserstein, not full multidimensional optimal transport.
    """
    _check_pair(x, y)
    if len(x) != len(y):
        raise ValueError("Exact sort-based SW here requires equal sample counts")
    if num_slices < 1:
        raise ValueError("num_slices must be positive")
    with torch.autocast(device_type=x.device.type, enabled=False):
        generator = torch.Generator(device=x.device).manual_seed(int(seed))
        directions = F.normalize(torch.randn(
            x.shape[1], num_slices, device=x.device, generator=generator,
            dtype=torch.float32), dim=0)
        sx = (x.float() @ directions).sort(dim=0).values
        ty = (y.detach().float() @ directions).sort(dim=0).values
        return (sx - ty).square().mean()


def gather_batch(x, with_grad):
    """Gather equal-size local batches. Do not multiply the loss by world_size.

    Autograd all_gather sums backward contributions; DDP then averages parameter
    gradients, recovering the gradient of the single global-batch objective.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x if with_grad else x.detach()
    count = torch.tensor([len(x)], device=x.device, dtype=torch.long)
    counts = [torch.empty_like(count) for _ in range(dist.get_world_size())]
    dist.all_gather(counts, count)
    if any(c.item() != count.item() for c in counts):
        raise ValueError("Use equal local batch sizes on every rank (drop_last=True)")
    if with_grad:
        return torch.cat(grad_all_gather(x.contiguous()), dim=0)
    outputs = [torch.empty_like(x) for _ in counts]
    dist.all_gather(outputs, x.detach().contiguous())
    return torch.cat(outputs, dim=0)


def pool_eos(hidden, attention_mask):
    """Unnormalized EOS pooling, including expanded image-token sequences.

    Matches MMEBModel's trailing-padding convention, per row. It also handles
    left padding without incorrectly treating it as trailing padding.
    """
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected hidden [B, L, D] and attention_mask [B, T]")
    if hidden.shape[0] != attention_mask.shape[0] or not attention_mask.bool().any(1).all():
        raise ValueError("Invalid or fully masked samples")
    trailing_padding = (~attention_mask.bool()).flip(1).cumprod(1).sum(1)
    indices = hidden.shape[1] - trailing_padding - 1
    if (indices < 0).any():
        raise ValueError("Attention-mask padding exceeds hidden sequence length")
    return hidden[torch.arange(len(hidden), device=hidden.device), indices]


class _DistributionKD(nn.Module):
    method = None

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.register_buffer("sw_step", torch.zeros((), dtype=torch.long))
        self.bandwidth = float(getattr(args, "mmd_bandwidth", 0.0))
        raw_scales = getattr(args, "mmd_kernel_scales", "0.5,1,2")
        self.scales = tuple(float(x) for x in raw_scales.split(","))
        if self.bandwidth < 0 or not math.isfinite(self.bandwidth):
            raise ValueError("mmd_bandwidth must be finite and >= 0")
        if not self.scales or any(s <= 0 or not math.isfinite(s) for s in self.scales):
            raise ValueError("mmd_kernel_scales must be finite and positive")
        weight = getattr(args, "distribution_weight", None)
        weight = getattr(args, "sigreg_weight", 0.05) if weight is None else weight
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("distribution_weight must be finite and >= 0")
        self.num_slices = int(getattr(args, "distribution_num_slices", 256))
        if self.num_slices < 1:
            raise ValueError("distribution_num_slices must be positive")

    @staticmethod
    def _teacher_reps(batch, side, student_reps):
        caches = batch.get(f"teacher_{side}_caches")
        if caches is None or len(caches) != len(student_reps):
            raise ValueError(f"{side}: supply --caching_dir with one teacher rep per student sample")
        reps = []
        for i, cache in enumerate(caches):
            rep = cache.get("rep") if isinstance(cache, dict) else cache
            if not torch.is_tensor(rep) or rep.ndim != 1 or not torch.isfinite(rep).all():
                raise ValueError(f"{side} sample {i}: cache rep must be a finite vector [D_teacher]")
            reps.append(rep.detach().to(device=student_reps.device, dtype=torch.float32))
        return torch.stack(reps)

    def _common_space(self, student, teacher, wrapper):
        """Only student is projected; never learn a transform of teacher targets."""
        projectors = getattr(wrapper, "projectors", None)
        if projectors is not None and "s2t" in projectors:
            projector = projectors["s2t"]
            # F.linear in fp32 avoids BF16 input/weight mismatches and loss of
            # small covariance differences. Gradient still reaches BF16 weights.
            for layer in projector if isinstance(projector, nn.Sequential) else [projector]:
                if not isinstance(layer, nn.Linear):
                    raise ValueError("Use config/projector_config_distribution.json (linear s2t)")
                student = F.linear(student.float(), layer.weight.float(),
                                   None if layer.bias is None else layer.bias.float())
        elif student.shape[1] != teacher.shape[1]:
            raise ValueError("Different dimensions require --projector_config_path config/projector_config_distribution.json")
        _check_pair(student, teacher)
        if getattr(self.args, "distribution_normalize", False):
            student, teacher = F.normalize(student, dim=-1), F.normalize(teacher, dim=-1)
        return student, teacher

    def distribution_loss(self, student, teacher):
        if self.method == "mmd":
            return mmd_loss(student, teacher, self.bandwidth, self.scales)
        if self.method == "coral":
            return coral_loss(student, teacher)
        seed = int(getattr(self.args, "seed", 42)) + int(self.sw_step.item())
        result = sliced_wasserstein_loss(student, teacher, self.num_slices, seed)
        if self.training:
            self.sw_step.add_(1)
        return result

    def forward(self, model_wrapper, input_data):
        model = model_wrapper.model
        qout = model.encode_input(input_data["qry"])
        pout = model.encode_input(input_data["pos"])
        qrep, prep = qout[0], pout[0]
        if len(qrep) != len(prep):
            raise ValueError("These baselines require exactly one positive per query")
        qglobal, pglobal = gather_batch(qrep, True), gather_batch(prep, True)
        scores = model.compute_similarity(qglobal, pglobal).float()
        temperature = float(model_wrapper.temperature)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        labels = torch.arange(len(qglobal), device=scores.device)
        contrastive = F.cross_entropy(scores / temperature, labels)
        structural, distribution = contrastive.new_zeros(()), contrastive.new_zeros(())
        use_struct = getattr(self.args, "use_distill_loss", True)
        use_dist = getattr(self.args, "use_sigreg_loss", True)
        if use_struct or use_dist:
            tq = self._teacher_reps(input_data, "qry", qrep)
            tp = self._teacher_reps(input_data, "pos", prep)
            teacher = gather_batch(torch.cat((tq, tp)), False)
            # Gather pooled vectors only; do NOT gather token sequences. Token
            # lengths and padding legitimately differ across DDP ranks.
            with torch.autocast(device_type=qrep.device.type, enabled=False):
                if use_struct:
                    sq = pool_eos(qout[3][-1], input_data["qry"]["attention_mask"])
                    sp = pool_eos(pout[3][-1], input_data["pos"]["attention_mask"])
                    student = gather_batch(torch.cat((sq, sp)).float(), True)
                    ns, nt = F.normalize(student, dim=-1), F.normalize(teacher, dim=-1)
                    structural = F.mse_loss(ns @ ns.T, nt @ nt.T)
                if use_dist:
                    layer = int(getattr(self.args, "num_layers", -1))
                    sq = pool_eos(qout[3][layer], input_data["qry"]["attention_mask"])
                    sp = pool_eos(pout[3][layer], input_data["pos"]["attention_mask"])
                    local_student, local_teacher = self._common_space(
                        torch.cat((sq, sp)).float(), torch.cat((tq, tp)), model_wrapper)
                    student = gather_batch(local_student, True)
                    # teacher here must use the same common-space transform.
                    target = gather_batch(local_teacher, False)
                    distribution = self.distribution_loss(student, target)
        weight = getattr(self.args, "distribution_weight", None)
        if weight is None:
            weight = getattr(self.args, "sigreg_weight", 0.05)
        loss = contrastive + self.args.kd_weight * structural + weight * distribution
        return {"loss": loss, "contrastive_loss": contrastive, "kd_loss": structural,
                "sigreg_loss": distribution, "distribution_loss": distribution}


class MMDKDLoss(_DistributionKD):
    method = "mmd"


class CORALKDLoss(_DistributionKD):
    method = "coral"


class SWKDLoss(_DistributionKD):
    """Direct sliced W2^2 after s2t projection in the teacher space."""
    method = "sw"
