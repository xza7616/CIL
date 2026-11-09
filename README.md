# CIL

RC2GS 与 component-wise Scale-Causal GRPO 分割实验的代码快照。

## 项目内容

- `train_rc2gs_gcr_grpo.py`：训练入口。
- `evaluate.py`：评估脚本。
- `configs/`：SUIM 数据集及训练配置。
- `toolbox/`：数据处理、模型、损失、指标及 DINOv3 相关代码。
- `KD_loss/`：知识蒸馏相关损失。

## 使用前准备

本仓库按当前本地项目内容上传，尚未验证可独立训练。训练入口引用的
`train_rc2gs_sar.py` 和 `toolbox/models/text4/DINO_RC2GS_SAR.py`
目前不在项目源码中，需要补齐相关依赖后再运行。

请根据实际环境安装依赖，并将 `configs/*.json` 中的 `root` 修改为本地数据集路径。
DINOv3 的说明和依赖文件位于 `toolbox/DINO/dinov3/`。

## 预训练权重

以下本地权重未纳入 Git，使用时请单独准备并放入 `toolbox/DINO/`：

- `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
- `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth`
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

Python 缓存、训练输出和模型权重由根目录 `.gitignore` 排除。

## 第三方代码

随附 DINOv3 代码保留其原有文档和许可证，具体条款见
[`toolbox/DINO/dinov3/LICENSE.md`](toolbox/DINO/dinov3/LICENSE.md)。
