# Simulator Setup and Evaluation

Bridge-WA evaluation uses a policy-server plus simulator-client workflow.

1. The server runs in the `bridge-wa` environment and loads a Bridge-WA checkpoint.
2. The simulator client runs in the benchmark environment.
3. The client sends observations and language instructions to the server over HTTP and executes returned action chunks.

## VLABench

The VLABench code is vendored under `evaluation/vlabench/VLABench`. Large simulator assets are not included.

Install:

```bash
conda create -n vlabench python=3.10 -y
conda activate vlabench
cd evaluation/vlabench/VLABench
pip install -r requirements.txt
pip install -e .
cd /path/to/bridge-wa
bash scripts/download_vlabench_assets.sh
```

The VLABench asset downloader should populate:

- `evaluation/vlabench/VLABench/VLABench/assets/obj/`
- `evaluation/vlabench/VLABench/VLABench/assets/scenes/`

Run Bridge-WA evaluation:

```bash
cd /path/to/bridge-wa
MODEL_PATH=./checkpoints/bridge_wa_vlabench \
VLABENCH_CONDA_ENV=vlabench \
SERVER_CONDA_ENV=bridge-wa \
bash scripts/eval_bridge_wa_vlabench.sh
```

Default official protocol:

- Tracks: `track_1_in_distribution`, `track_2_cross_category`, `track_3_common_sense`, `track_4_semantic_instruction`
- Metrics: `success_rate`, `intention_score`, `progress_score`
- Episodes per track: `10`

Useful overrides:

```bash
RUN_IN_BACKGROUND=0 STOP_SERVER_AFTER_EVAL=1 bash scripts/eval_bridge_wa_vlabench.sh
RUN_CLIENT=0 bash scripts/eval_bridge_wa_vlabench.sh
EVAL_TRACKS="track_6_unseen_texture" STRICT_BRIDGE_WA_OFFICIAL=0 bash scripts/eval_bridge_wa_vlabench.sh
```
