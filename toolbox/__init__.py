from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, adjust_lr
from .ranger.ranger import Ranger
from .ranger.ranger913A import RangerVA
from .ranger.rangerqh import RangerQH

def get_dataset(cfg):
    assert cfg['dataset'] in ['SUIM', 'WE3Ds', 'UPLight']

    # if cfg['dataset'] == 'nyuv2':
    #     from .datasets.SUIM import NYUv2
    #     return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    # if cfg['dataset'] == 'sunrgbd':
    #     from .datasets.sunrgbd import SUNRGBD
    #     return SUNRGBD(cfg, mode='train'), SUNRGBD(cfg, mode='test')
    if cfg['dataset'] == 'WE3Ds':
        from .datasets.WE3Ds import WE3Ds
        return WE3Ds(cfg, mode='train'), WE3Ds(cfg, mode='test')
    if cfg['dataset'] == 'SUIM':
        from .datasets.SUIM import SUIM
        return SUIM(cfg, mode='train'), SUIM(cfg, mode='test')
    if cfg['dataset'] == 'UPLight':
        from .datasets.UPLight_pga import UPLight
        return UPLight(cfg, mode='train'), UPLight(cfg, mode='test')


def get_model(cfg):
    if cfg['model_name'] == 'DINO_RC2GS_C3A':
        from toolbox.models.text4.DINO_RC2GS_C3A import build_model
        return build_model(cfg)
    if cfg['model_name'] == 'DINO_RC2GS_SCRA':
        from toolbox.models.text4.DINO_RC2GS_SCRA import build_model
        return build_model(cfg)
    if cfg['model_name'] == 'DINO_RC2GS_CGVA_ABLATION':
        from toolbox.models.text4.DINO_RC2GS_CGVA_ABLATION import build_model
        return build_model(cfg)
    if cfg['model_name'] == 'DINO_RC2GS_CGVA':
        from toolbox.models.text4.DINO_RC2GS_CGVA import EncoderDecoder
        return EncoderDecoder(
            num_classes=cfg.get('n_classes', 8),
            freeze_backbone=cfg.get('freeze_backbone', True),
            stride=cfg.get('stride', 16),
            feature_dim=cfg.get('feature_dim', 256),
            gaussian_kernel_size=cfg.get('gaussian_kernel_size', 3),
            gaussian_query_dim=cfg.get('gaussian_query_dim', 32),
            local_router_hidden=cfg.get('local_router_hidden', 64),
            local_blend_max=cfg.get('local_blend_max', 0.35),
            local_balance_weight=cfg.get('local_balance_weight', 0.01),
            dino_repo=cfg.get('dino_repo'),
            dino_weights=cfg.get('dino_weights'),
            agent_grid_size=cfg.get('agent_grid_size', 4),
            agent_num_proposals=cfg.get('agent_num_proposals', 3),
            agent_max_steps=cfg.get('agent_max_steps', 2),
            agent_hidden_dim=cfg.get('agent_hidden_dim', 160),
            agent_blend_max=cfg.get('agent_blend_max', 0.50),
            agent_mask_sigma=cfg.get('agent_mask_sigma', 0.58),
            agent_credit_scale=cfg.get('agent_credit_scale', 0.50),
            agent_verify_threshold=cfg.get('agent_verify_threshold', 0.002),
            agent_min_confidence=cfg.get('agent_min_confidence', 0.50),
            agent_verifier_route_bias=cfg.get('agent_verifier_route_bias', 0.40),
        )
    if cfg['model_name'] == 'DINO_RC2GS_AGENT':
        from toolbox.models.text4.DINO_RC2GS_AGENT import EncoderDecoder
        return EncoderDecoder(
            num_classes=cfg.get('n_classes', 8),
            freeze_backbone=cfg.get('freeze_backbone', True),
            stride=cfg.get('stride', 16),
            feature_dim=cfg.get('feature_dim', 256),
            gaussian_kernel_size=cfg.get('gaussian_kernel_size', 3),
            gaussian_query_dim=cfg.get('gaussian_query_dim', 32),
            local_router_hidden=cfg.get('local_router_hidden', 64),
            local_blend_max=cfg.get('local_blend_max', 0.35),
            local_balance_weight=cfg.get('local_balance_weight', 0.01),
            dino_repo=cfg.get('dino_repo'),
            dino_weights=cfg.get('dino_weights'),
            agent_grid_size=cfg.get('agent_grid_size', 3),
            agent_num_proposals=cfg.get('agent_num_proposals', 2),
            agent_max_steps=cfg.get('agent_max_steps', 2),
            agent_blend_strength=cfg.get('agent_blend_strength', 0.8),
            agent_hidden_dim=cfg.get('agent_hidden_dim', 128),
            agent_credit_scale=cfg.get('agent_credit_scale', 0.25),
        )
    comparison_models = {
        'CMNeXt_CVPR2023',
        'DFormer_ICLR2024',
        'GeminiFusion_ICML2024',
        'PrimKD_ACMMM2024',
        'DFormerv2_CVPR2025',
        'KTB_CVPR2025',
        'DiffPixelFormer_TCSVT2026',
    }
    if cfg['model_name'] in comparison_models:
        from toolbox.models.comparison_official import build_model
        crop_size = tuple(int(value) for value in cfg.get('crop_size', '480 640').split())
        return build_model(
            cfg['model_name'],
            num_classes=cfg.get('n_classes', 8),
            variant=cfg.get('official_variant'),
            pretrained=cfg.get('pretrained'),
            image_size=crop_size,
        )
    if cfg['model_name'] == 'DINO_RC2GS_SAR_ABLATION':
        from toolbox.models.text4.DINO_RC2GS_SAR_ABLATION import EncoderDecoder
        return EncoderDecoder(
            num_classes=cfg.get('n_classes', 8),
            freeze_backbone=cfg.get('freeze_backbone', True),
            stride=cfg.get('stride', 16),
            feature_dim=cfg.get('feature_dim', 256),
            gaussian_kernel_size=cfg.get('gaussian_kernel_size', 3),
            gaussian_query_dim=cfg.get('gaussian_query_dim', 32),
            local_router_hidden=cfg.get('local_router_hidden', 64),
            local_blend_max=cfg.get('local_blend_max', 0.35),
            local_balance_weight=cfg.get('local_balance_weight', 0.01),
            field_mode=cfg.get('field_mode', 'sar'),
            dino_repo=cfg.get('dino_repo'),
            dino_weights=cfg.get('dino_weights'),
        )
    if cfg['model_name'] == 'DINO_RC2GS_GCR_GRPO':
        from toolbox.models.text4.DINO_RC2GS_GCR_GRPO import EncoderDecoder
        return EncoderDecoder(
            num_classes=cfg.get('n_classes', 8),
            freeze_backbone=cfg.get('freeze_backbone', True),
            stride=cfg.get('stride', 16),
            feature_dim=cfg.get('feature_dim', 256),
            gaussian_kernel_size=cfg.get('gaussian_kernel_size', 3),
            gaussian_query_dim=cfg.get('gaussian_query_dim', 32),
            local_router_hidden=cfg.get('local_router_hidden', 64),
            local_blend_max=cfg.get('local_blend_max', 0.35),
            local_balance_weight=cfg.get('local_balance_weight', 0.01),
            dino_repo=cfg.get('dino_repo'),
            dino_weights=cfg.get('dino_weights'),
        )
    if cfg['model_name'] == 'DINO_RC2GS_SAR':
        from toolbox.models.text4.DINO_RC2GS_SAR import EncoderDecoder
        return EncoderDecoder(
            num_classes=cfg.get('n_classes', 8),
            freeze_backbone=cfg.get('freeze_backbone', True),
            stride=cfg.get('stride', 16),
            feature_dim=cfg.get('feature_dim', 256),
            gaussian_kernel_size=cfg.get('gaussian_kernel_size', 3),
            gaussian_query_dim=cfg.get('gaussian_query_dim', 32),
            local_router_hidden=cfg.get('local_router_hidden', 64),
            local_blend_max=cfg.get('local_blend_max', 0.35),
            local_balance_weight=cfg.get('local_balance_weight', 0.01),
            dino_repo=cfg.get('dino_repo'),
            dino_weights=cfg.get('dino_weights'),
        )
    if cfg['model_name'] == 'SUMENet':
        from toolbox.DINO.SUMENet import EncoderDecoder
        return EncoderDecoder(num_classes=8, freeze_backbone=True, topk=4)
    if cfg['model_name'] == 'DINOv3':
        from toolbox.DINO.DINOv3 import EncoderDecoder
        return EncoderDecoder()
    raise ValueError("Unknown model_name: {}".format(cfg['model_name']))
