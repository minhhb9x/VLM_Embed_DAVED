from src.criterions.contrastive_loss import ContrastiveLoss
from .daved import DAVED

criterion_list = {
    "contrastive": ContrastiveLoss,
    "daved": DAVED,
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)