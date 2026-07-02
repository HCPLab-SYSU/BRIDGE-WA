<h1 align="center">Bridge-WA: Predicting Where and How the World Changes for Robotic Action</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=W_YDucAAAAAJ&hl=zh-CN">Yongjie Bai</a>,
  Hanting Wang,
  <a href="https://github.com/CCCalcifer/">Mingtong Dai</a>,
  Qijun Zhong,
  <a href="https://yangliu9208.github.io/">Yang Liu</a>,
  <a href="http://www.linliang.net/">Liang Lin</a>
</p>

<p align="center">
  <a href="https://hcplab-sysu.github.io/BRIDGE-WA">
    <img src="https://img.shields.io/badge/Project_Page-1f6feb?logo=google-chrome&logoColor=white&style=flat-square" alt="Project Page" />
  </a>
  <a href="https://huggingface.co/baiyu858/Bridge-WA-VLABench">
    <img src="https://img.shields.io/badge/HuggingFace-Bridge--WA--VLABench-f59e0b?logo=huggingface&logoColor=white&style=flat-square" alt="HuggingFace Bridge-WA VLABench" />
  </a>
  <a href="assets/docs/Bridge-WA.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-b31b1b?logo=adobeacrobatreader&logoColor=white&style=flat-square" alt="Paper PDF" />
  </a>
</p>

<p align="center">
  <video src="assets/videos/BridgeWA_release_video_v4.mp4" controls muted loop width="100%"></video>
</p>

**Bridge-WA** distills a frozen **World Teacher** into compact world-change priors for robotic action. It predicts **Future Tokens**, **Change Maps**, and **Motion-Flow Maps**, then conditions lightweight action-transformer blocks for policy learning.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Clipboard/3D/clipboard_3d.png" width="28" alt="Clipboard" /> Table of Contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Model Zoo](#model-zoo)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Sparkles/3D/sparkles_3d.png" width="28" alt="Sparkles" /> Highlights

- **World Teacher** pre-training and post-training scripts.
- Cache precomputation for Future Tokens, Change Maps, and Motion-Flow Maps.
- Bridge-WA training and online evaluation on **VLABench**.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Gear/3D/gear_3d.png" width="28" alt="Gear" /> Installation

### 1. Create the Bridge-WA environment

```bash
conda env create -f environment.yml
conda activate bridge-wa
pip install -r requirements.txt
export PYTHONPATH=$PWD:${PYTHONPATH}
```

### 2. Install simulator environments

Bridge-WA evaluation uses a policy-server plus simulator-client workflow. The policy server runs in the `bridge-wa` environment; the VLABench client runs in its own benchmark environment.

See:

- `evaluation/README.md`
- `evaluation/vlabench/README.md`

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Package/3D/package_3d.png" width="28" alt="Package" /> Model Zoo

| Model | Hugging Face |
| --- | --- |
| World Teacher pretrained on BridgeData V2 | [baiyu858/Bridge-WA-World-Teacher-BridgedataV2](https://huggingface.co/baiyu858/Bridge-WA-World-Teacher-BridgedataV2) |
| World Teacher post-trained on VLABench | [baiyu858/Bridge-WA-World-Teacher-VLABench](https://huggingface.co/baiyu858/Bridge-WA-World-Teacher-VLABench) |
| World Teacher post-trained on RoboTwin2.0 | [baiyu858/Bridge-WA-World-Teacher-RoboTwin](https://huggingface.co/baiyu858/Bridge-WA-World-Teacher-RoboTwin) |
| Bridge-WA policy trained on VLABench | [baiyu858/Bridge-WA-VLABench](https://huggingface.co/baiyu858/Bridge-WA-VLABench) |

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Rocket/3D/rocket_3d.png" width="28" alt="Rocket" /> Training Pipeline

### 1. Post-train the World Teacher

| Command | Purpose |
| --- | --- |
| `bash scripts/posttrain_world_teacher_vlabench.sh` | VLABench World Teacher post-training |

### 2. Precompute Bridge-WA cache

```bash
bash scripts/precompute_bridge_wa_cache_vlabench.sh
```

The cache stores Future Token, Change Map, and Motion-Flow targets generated from the frozen World Teacher.

### 3. Train Bridge-WA

| Command | Benchmark |
| --- | --- |
| `bash scripts/train_bridge_wa_vlabench.sh` | VLABench |

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Rocket/3D/rocket_3d.png" width="28" alt="Rocket" /> Evaluation

### VLABench

```bash
MODEL_PATH=./checkpoints/bridge_wa_vlabench \
VLABENCH_CONDA_ENV=vlabench \
SERVER_CONDA_ENV=bridge-wa \
bash scripts/eval_bridge_wa_vlabench.sh
```

The script starts `scripts/deploy_bridge_wa.py` as a local policy server and then launches the benchmark client.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Package/3D/package_3d.png" width="28" alt="Package" /> Repository Layout

```text
bridge-wa/
  assets/                         paper, project-page, and video assets
  datasets/                       dataset handlers and domain adapters
  evaluation/                     VLABench evaluation client and benchmark setup
  models/                         World Teacher, Vision Action, and Bridge-WA modules
  scripts/                        training, cache, deployment, and evaluation scripts
  scripts/train_world_teacher.py  World Teacher training entry
  scripts/train_bridge_wa.py      Bridge-WA training entry and world-guidance utilities
  index.html                      project page
```

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Books/3D/books_3d.png" width="28" alt="Books" /> Citation

If you find **Bridge-WA** useful in your research, please cite:

```bibtex
@article{bai2026bridgewa,
  title   = {Bridge-WA: Predicting Where and How the World Changes for Robotic Action},
  author  = {Bai, Yongjie and Wang, Hanting and Dai, Mingtong and Zhong, Qijun and Liu, Yang and Lin, Liang},
  year    = {2026}
}
```

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Sparkles/3D/sparkles_3d.png" width="28" alt="Sparkles" /> Acknowledgements

This repository builds on several open-source projects and benchmark environments, including [X-VLA](https://github.com/2toinf/X-VLA.git), [Wan2.2](https://github.com/Wan-Video/Wan2.2.git), [VLABench](https://github.com/OpenMOSS/VLABench.git), [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin.git).
