# Simulator Setup and Evaluation

Bridge-WA evaluation uses a policy-server plus simulator-client workflow.

1. The server runs in the `bridge-wa` environment and loads a Bridge-WA checkpoint.
2. The simulator client runs in the benchmark environment.
3. The client sends observations and language instructions to the server over HTTP and executes returned action chunks.

## VLABench

The VLABench code is vendored under `evaluation/vlabench/VLABench`. Large simulator assets are not included.

Install:

```bash
cd /path/to/bridge-wa
conda create -n vlabench python=3.10 -y
conda run -n vlabench python -m pip install -e "evaluation/vlabench/VLABench[bridge-wa-eval]"
bash scripts/download_vlabench_assets.sh
```

Use the `bridge-wa-eval` installation extra for policy evaluation. The vendored
VLABench `requirements.txt` also installs its training, dataset conversion,
notebook, and alternate-policy dependencies, including Git-based LeRobot and
scripted RRT packages that are not needed by the Bridge-WA client.

To repair an existing partial `vlabench` environment, rerun the `pip` command
above. It installs the tested simulator pins and the imports needed by the
Bridge-WA evaluation client.

The VLABench asset downloader should populate:

- `evaluation/vlabench/VLABench/VLABench/assets/base/`
- `evaluation/vlabench/VLABench/VLABench/assets/obj/`
- `evaluation/vlabench/VLABench/VLABench/assets/robots/franka_emika_panda/`
- `evaluation/vlabench/VLABench/VLABench/assets/scenes/`

`base/` and `robots/franka_emika_panda/` are MuJoCo simulation files, not
real-robot drivers. Current upstream VLABench archives omit these directories,
so the downloader restores them from the last official Git revision that
tracked them.

Run Bridge-WA evaluation:

```bash
cd /path/to/bridge-wa
MODEL_PATH=./checkpoints/bridge_wa_vlabench \
VLABENCH_CONDA_ENV=vlabench \
SERVER_CONDA_ENV=bridge-wa \
bash scripts/eval_bridge_wa_vlabench.sh
```

Default official protocol:

- Tracks: `track_1_in_distribution`, `track_2_cross_category`, `track_3_common_sense`, `track_4_semantic_instruction`, `track_6_unseen_texture`
- Metrics: `success_rate`, `intention_score`, `progress_score`
- Episodes per task: `10` (`100` episodes per track, `500` episodes total)
- Server port: `8000`

`N_EPISODE` is the number for every task, matching the paper protocol. `PORT`
can override the default for a single run without changing the repository
default. Video capture is disabled by default to keep long evaluations memory
bounded; users can set `SAVE_VIDEO=1` to enable streaming video output. The
older `VISUALIZATION` environment variable remains supported as an alias.

Useful overrides:

```bash
RUN_IN_BACKGROUND=0 STOP_SERVER_AFTER_EVAL=1 bash scripts/eval_bridge_wa_vlabench.sh
RUN_CLIENT=0 bash scripts/eval_bridge_wa_vlabench.sh
PREFLIGHT_ONLY=1 RUN_IN_BACKGROUND=0 bash scripts/eval_bridge_wa_vlabench.sh
RESUME=1 RUN_IN_BACKGROUND=0 bash scripts/eval_bridge_wa_vlabench.sh
SAVE_VIDEO=1 RUN_IN_BACKGROUND=0 bash scripts/eval_bridge_wa_vlabench.sh
N_EPISODE=1 STRICT_BRIDGE_WA_OFFICIAL=0 bash scripts/eval_bridge_wa_vlabench.sh
EVAL_TRACKS="track_6_unseen_texture" STRICT_BRIDGE_WA_OFFICIAL=0 bash scripts/eval_bridge_wa_vlabench.sh
```

The launcher checks the simulator imports and requested server port before it
loads the model. On failure it stops the server it started; after a successful
run, `STOP_SERVER_AFTER_EVAL=1` also stops it instead of leaving it available
for another client.
