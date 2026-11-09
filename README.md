# CIL
。

## 项目内容

- `train_rc2gs_gcr_grpo.py`：训练入口。
- `evaluate.py`：评估脚本。
- `configs/`：SUIM 数据集及训练配置。
- `toolbox/`：数据处理、模型、损失、指标及 DINOv3 相关代码

## 预训练权重

以下本地权重未纳入 Git，使用时请单独准备并放入 `toolbox/DINO/`：

- `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
- `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth`
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

