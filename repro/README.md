# CIFAR-N 交叉熵（CE）基线复现记录

运行环境：4 核 CPU（无 GPU），PyTorch 2.14.1（CPU 运行），NumPy 2.5。

## 1. 数据集统计（精确复现）

噪声率 = 噪声标签与 `clean_label` 不一致的比例，由 `data/CIFAR-*_human.pt` 计算：

| 标签集 | 本次计算 | 论文 Table 1 |
|---|---|---|
| CIFAR-10N aggre | 9.01% | 9.03% |
| CIFAR-10N random1 | 17.23% | 17.23% |
| CIFAR-10N random2 | 18.12% | 18.12% |
| CIFAR-10N random3 | 17.64% | 17.64% |
| CIFAR-10N worst | 40.21% | 40.21% |
| CIFAR-100N noisy | 40.20% | 40.20% |

aggre 相差 0.02%，可能是因为仓库后来更新过标签文件。

## 2. 训练结果（缩减预算）

设置：ResNet-18，10 个 epoch（前 6 个 lr=0.1，后 4 个 lr=0.01），SGD momentum 0.9，weight decay 5e-4，batch 128，
数据增强与原代码相同，seed 0，每个设置跑 1 次。论文设置为 ResNet-34、100 个 epoch（前 60 个 lr=0.1，后 40 个 lr=0.01）。

| 设置 | 噪声率 | 本次（最后一个 epoch） | 论文 CE | 相对 clean 的掉点：本次 / 论文 |
|---|---|---|---|---|
| CIFAR-10N clean | 0% | 86.00 | — | — |
| CIFAR-10N aggre | 9.01% | 84.11 | 87.77 | −1.9 / 约 −5 |
| CIFAR-10N worst | 40.21% | 72.10 | 77.69 | −13.9 / 约 −15 |
| CIFAR-100N noisy | 40.20% | 51.29 | 55.50 | — |

论文 CE 数值取自 CIFAR-N 排行榜（SplitNet 论文 arXiv:2211.11753 中的转载）。本环境访问不到论文原文，所以 clean 列没有填；“论文掉点”一列按论文 clean 约 92.9 估算，这个数值来自记忆，未经核实。
4 个设置都是最后一个 epoch 准确率最高（best = last），逐 epoch 曲线见 `logs/`。

### 结论
- 准确率排序 clean > aggre > worst 与论文一致，worst（40% 人工噪声）掉点幅度也与论文接近。
- 绝对数值比论文低 4~6 个点，主要因为模型更小、训练轮数只有论文的 1/10。
- aggre 掉点比论文小，很可能是因为只训练 10 个 epoch、降学习率后只有 4 个 epoch 时，网络还没来得及记住噪声标签；
  论文的 100 个 epoch 会在降学习率后的长阶段里拟合噪声，掉点因此更大。
- 本次没有跑 rand1/2/3、CIFAR-100N clean，也没有跑合成噪声对照（不加 `--is_human`）。

## 3. 复现步骤

```bash
# 1) 准备数据（cs.toronto.edu 不可达时，从任意镜像获取官方二进制版 cifar-10-binary.tar.gz / cifar-100-binary.tar.gz 并解压）
python repro/prepare_cifar_from_binary.py --c10 <cifar-10-batches-bin 目录> --c100 <cifar-100-binary 目录> --out ~/data
#    脚本会先核对训练集顺序（50000/50000 个标签需与 CIFAR-N 的 clean_label 一致）再写文件。
#    能访问官方下载地址的机器跳过这一步即可，main.py 会自动下载。

# 2) 本次的缩减版（CPU 约 1h50m/设置；可断点续跑，已完成的设置自动跳过）
repro/run_reduced.sh
python repro/summarize.py

# 3) 论文完整设置（GPU；ResNet-34、100 epoch）
MODEL=ResNet34 N_EPOCH=100 EXTRA="--num_workers 4" \
RUNS="cifar10:clean cifar10:aggre cifar10:rand1 cifar10:rand2 cifar10:rand3 cifar10:worst cifar100:clean100 cifar100:noisy100" \
repro/run_reduced.sh
#   或直接：python main.py --dataset cifar10 --noise_type worst --is_human
```

## 4. 对原代码的修改

- `main.py`：修复 `evaluate()` 中未定义的 `best_acc_`（原代码第一个 epoch 结束就会抛 NameError）；自动选择 cuda/cpu；
  去掉已废弃的 `reduce=True`；评估时用 `no_grad`；学习率计划按 `--n_epoch` 缩放（默认 100 epoch 时与原来完全相同）；
  新增 `--model`、`--channels_last`、`--ckpt`（每个 epoch 存档，支持续跑）。
- `data/cifar.py`：`torch.load(..., weights_only=False)`（PyTorch ≥ 2.6 必需）；允许 md5 不一致但文件存在的数据目录。
- `data/utils.py`：`np.where(...)[0][0]`，修复 NumPy 2 下合成噪声分支报错（随机数抽取不变，生成的合成标签与原来一致）。
