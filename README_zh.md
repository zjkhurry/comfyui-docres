# comfyui-docres

ComfyUI 的文档图像修复节点 —— 把弯曲的页面拉平、去掉阴影和模糊、输出干净的黑白扫描件。

包含两条互相独立的展平路线：一条全自动，另一条的控制点可以在展平前手动校正，用于自动结果不够好的情况。

![原始扫描图与修复结果对比](imgs/1.png)

全部只做推理，仓库里不含训练代码。

## 能做什么

| 任务 | 效果 |
|------|------|
| **展平（Dewarp）** | 把弯曲或折叠的页面还原成矩形 |
| **去阴影（Deshadow）** | 去掉阴影、渐变和光照不均 |
| **外观修复（Appearance）** | 修对比度、背面透印和色偏 |
| **去模糊（Deblur）** | 锐化运动模糊和失焦模糊 |
| **二值化（Binarize）** | 输出干净的黑白二值图 |

前四项来自同一个模型（[DocRes](https://arxiv.org/abs/2405.04408)），一个模型覆盖这四种任务。二值化是另外的传统算法路线。

## 用了什么技术

**DocRes** —— 用一个 Restormer 主干覆盖全部四项学习任务。任务信息通过一张 3 通道的 *prompt* 图传给网络 —— 这张图用廉价的传统 CV 从输入图算出，再和 RGB 三通道拼接。所以同一份权重不做重训就能干五件事。

**DDC 控制点展平** —— [Document-Dewarping-with-Control-Points](https://arxiv.org/abs/2203.10543) 的移植。网络预测出一张铺在页面上的控制点网格；因为这些点就是普通数字，你可以在展平之前把它们拖到想要的位置。展平本身是对这个网格拟合薄板样条。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zjkhurry/comfyui-docres.git
pip install -r comfyui-docres/requirements.txt
```

权重（约 335 MB）**不在本仓库里** —— 节点第一次用到时会自动从 [Hugging Face](https://huggingface.co/zjkhurry/comfyui-docres-weights) 下载，并缓存到 `comfyui-docres/weights/`。不需要做任何操作。

| 权重 | 大小 | 用途 |
|------|------|------|
| `docres.safetensors` | 58 MB | DocRes Restore |
| `mbd.safetensors` | 227 MB | DocRes Restore —— 仅 `dewarping` 用 |
| `ddc_fiducial1024_v1.safetensors` | 51 MB | DDC Predict Points |

如果 `ComfyUI/models/` 下已经有权重，会直接使用、不会重复下载。

重启 ComfyUI，所有节点出现在 **image/DocRes** 分类下。

## 节点说明

### DocRes Loader

加载一次权重，输出一个 `model` 句柄。一个 loader 可以喂整条链，所以无论串多少任务，权重都**只从磁盘读一次**。

![DocRes Loader](imgs/6.png)

- **`device`** —— `auto`（默认），或强制 `cuda` / `mps` / `cpu`

### DocRes Restore

用这个句柄执行其中一项任务。

![DocRes Restore](imgs/7.png)

- **`image`** —— 待修复的扫描图
- **`task`** —— `dewarping`、`deshadowing`、`appearance`、`deblurring`、`binarization` 之一
- **`model`** —— 来自 DocRes Loader

连线方式：

```
LoadImage ─┬─▶ DocRes Restore (dewarping) ─▶ DocRes Restore (deshadowing) ─▶ DocRes Restore (appearance) ─▶ SaveImage
           │
DocRes Loader ── model ─┴──────────────────┴───────────────────────────────┘
```

### DDC Predict Points

跑控制点网络，输出控制点网格。

![DDC Predict Points](imgs/2.png)

- **`image`** —— 待修复的扫描图
- **`device`** —— `auto`（默认），或强制 `cuda` / `mps` / `cpu`

### DDC Edit Points

把控制点画在你的图上让你拖动。它从不跑网络，所以调整是即时且自由的。

![DDC Edit Points](imgs/3.png)

用 **edit control points** 按钮打开编辑器：

![控制点画布](imgs/4.png)

| 操作 | 方式 |
|------|------|
| 移动控制点 | 直接拖动 |
| 缩放 | `ctrl`/`cmd` + 滚轮，最大 12 倍 |
| 平移 | 滚轮，或 `空格`+拖动，或中键拖动，或右键拖动 |
| 复位视图 | **Reset view** 按钮 |
| 关闭 | `Esc` |

网络预测出 961 个点，逐个拖不现实，所以编辑器只编辑其中的 9×9 采样点，再拟合其余网格去匹配。你动过的点会变成黄色。

你的修改会保存在节点里，所以重跑下一个节点就能复现你的展平，保存的工作流也会把它们一起带走。

### DDC Rectify

对控制点拟合薄板样条，把页面展平。

![DDC Rectify](imgs/5.png)

- **`image`** —— 要接干净的 `image` 输出，**不要**接 `preview`，否则点会被烘进结果里

### 该用哪一个

DocRes 的展平是一次前向推理、不需要任何交互，所以先用它。如果自动结果把文字掰弯了或者跑出页面，再换控制点编辑器。两者互相独立，也可以和 DocRes 的其他任务串联使用。

```
LoadImage ─▶ DDC Predict Points ─▶ DDC Edit Points ─▶ DDC Rectify ─▶ SaveImage
```

## 致谢

移植自以下项目 —— 使用时请引用：

- **DocRes** — [A Generalist Model Toward Unifying Document Image Restoration Tasks](https://arxiv.org/abs/2405.04408)
- **Restormer** —— 共享主干（[论文](https://arxiv.org/abs/2111.09881)）
- **Document-Dewarping-with-Control-Points** —— 控制点展平器（[论文](https://arxiv.org/abs/2203.10543)、[代码](https://github.com/gwxie/Document-Dewarping-with-Control-Points)）
- **MBD** —— Mask-Based Deepwarping 分割，为 `dewarping` 任务提供页面掩膜（[代码](https://github.com/GuYuefei/MBD)）

编辑器面板的模态交互参考了 [ComfyUI-OpenPose-Studio](https://github.com/andreszs/ComfyUI-OpenPose-Studio)。

## 许可

移植代码遵循各上游项目的许可证。权重由原版转换后重新分发，具体条款请查阅对应上游仓库。
