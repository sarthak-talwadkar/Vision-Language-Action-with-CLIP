# Vision-Language-Action (VLA)

> **Status: Active Research — Work in Progress**
> This project is under active development. Architecture and results will be updated as experiments progress.

**Zero-shot robot action generation from natural language instructions and visual observations**, grounded in 3D scene understanding via CLIP-based visual-language alignment. The system takes an RGB image and a natural language instruction as input and outputs discrete action tokens executable in NVIDIA Isaac Sim — without task-specific training data for unseen instructions.

---

## Demo

![VLA Demo](assets/demo.gif)

> Input: RGB frame from Isaac Sim + natural language instruction. Output: discrete action token sequence controlling the robot. CLIP embeddings ground language to visual scene features before action prediction.

---

## Table of Contents

- [Motivation](#motivation)
- [System Architecture](#system-architecture)
- [Stage 1 — 3D Scene Understanding](#stage-1--3d-scene-understanding)
- [Stage 2 — CLIP Visual-Language Grounding](#stage-2--clip-visual-language-grounding)
- [Stage 3 — Action Generation (MLP Policy)](#stage-3--action-generation-mlp-policy)
- [Discrete Action Space](#discrete-action-space)
- [Isaac Sim Integration](#isaac-sim-integration)
- [Zero-Shot Generalization](#zero-shot-generalization)
- [Current Status](#current-status)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)

---

## Motivation

Classical robot manipulation pipelines require task-specific training data for every new instruction or object category. This severely limits deployability — a robot trained to "pick up the red cube" cannot generalize to "place the bottle near the bowl" without retraining.

**Vision-Language Models (VLMs)** like CLIP offer a different paradigm: they learn a shared embedding space for images and text from internet-scale data, enabling semantic grounding of novel instructions to visual observations without per-task supervision. The key insight this project exploits:

> If an image of a scene and a natural language instruction describing an action map to nearby points in CLIP embedding space, the alignment signal itself can guide action selection — without the model ever having seen that specific instruction during robot training.

The additional challenge addressed here is that **2D image features are insufficient for manipulation** — a robot operating in 3D space needs to understand object geometry, spatial relationships, and depth. This project integrates 3D scene understanding (OpenScene-style) with CLIP grounding to produce spatially-aware action tokens.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Inputs                                  │
│         RGB Image (Isaac Sim)   +   Natural Language Instruction│
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────┐    ┌─────────────────────────────┐
│   3D Scene Understanding │    │     CLIP Text Encoder       │
│                          │    │     (ViT-B/32 text tower)   │
│  RGB-D → point cloud     │    │                             │
│  OpenScene-style 3D      │    │  instruction → text         │
│  feature lifting         │    │  embedding  e_t ∈ R^512     │
│  → per-point CLIP        │    └──────────────┬──────────────┘
│    feature field         │                   │
│    F_3D ∈ R^{N×512}     │                   │
└──────────────┬───────────┘                   │
               │                               │
               ▼                               │
┌──────────────────────────┐                   │
│  CLIP Visual Encoder     │                   │
│  (ViT-B/32 image tower)  │                   │
│                          │                   │
│  RGB → image embedding   │                   │
│  e_v ∈ R^512             │                   │
└──────────────┬───────────┘                   │
               │                               │
               └──────────────┬────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │    Alignment + Fusion        │
               │                              │
               │  similarity = cos(e_v, e_t)  │
               │  fused = [e_v ‖ e_t ‖ F_3D] │
               │  → joint embedding R^{1536+} │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │     MLP Policy Head          │
               │                              │
               │  FC(1536 → 512) + ReLU       │
               │  FC(512 → 256)  + ReLU       │
               │  FC(256 → |A|)  + Softmax    │
               │                              │
               │  → discrete action token     │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │     Isaac Sim Executor       │
               │  action token → robot        │
               │  joint/gripper commands      │
               └──────────────────────────────┘
```

---

## Stage 1 — 3D Scene Understanding

Pure 2D CLIP features lack geometric grounding — they cannot represent object depth, spatial relationships between objects, or 3D affordances required for manipulation (e.g., "grasp from above" vs "slide from the side"). This stage lifts 2D CLIP features into 3D space via a point cloud representation.

**Pipeline:**

```
RGB-D frame (Isaac Sim)
    │
    ▼
Backproject depth → 3D point cloud  P = {p_i ∈ R³}
    │
    ▼
For each 3D point p_i:
    Project p_i → pixel (u_i, v_i) using camera intrinsics K
    Extract CLIP patch feature at (u_i, v_i) from image encoder
    Assign: F_3D[i] = CLIP_patch_feature(u_i, v_i)
    │
    ▼
Result: per-point CLIP feature field  F_3D ∈ R^{N × 512}
Each 3D point carries semantic CLIP features from the 2D image
```

**Why this matters for action generation:**

The 3D feature field enables spatial queries like:
```
"find the point cloud region most aligned with 'cup handle'"
→ argmax_i cos(F_3D[i], CLIP_text("cup handle"))
→ returns 3D location of handle → informs grasp approach direction
```

This gives the action policy geometric context that pure 2D CLIP embeddings cannot provide — the difference between knowing *what* an object is and knowing *where in 3D space* it is and *how to approach it*.

---

## Stage 2 — CLIP Visual-Language Grounding

CLIP (Contrastive Language-Image Pretraining) is trained to align image and text embeddings in a shared 512-dimensional space via contrastive learning on 400M image-text pairs. The key property exploited here: **semantic similarity in embedding space generalizes to unseen concepts** — a text description of an action not seen during training will still align to visually similar scenes seen during training.

**Encoding:**

```python
import clip
import torch

model, preprocess = clip.load("ViT-B/32", device="cuda")

# Visual encoding
image_input = preprocess(pil_image).unsqueeze(0).cuda()
with torch.no_grad():
    e_v = model.encode_image(image_input)        # R^512
    e_v = e_v / e_v.norm(dim=-1, keepdim=True)  # L2 normalize

# Language encoding
text_input = clip.tokenize(["pick up the red cube"]).cuda()
with torch.no_grad():
    e_t = model.encode_text(text_input)          # R^512
    e_t = e_t / e_t.norm(dim=-1, keepdim=True)  # L2 normalize

# Alignment score
similarity = (e_v * e_t).sum(dim=-1)  # cosine similarity ∈ [-1, 1]
```

**Fusion strategy:**

The visual embedding, text embedding, and aggregated 3D scene feature are concatenated to form the joint representation passed to the policy head:

```python
# Aggregate 3D features: max-pool over spatially relevant points
# (points within top-k similarity to text embedding)
topk_idx = torch.topk((F_3D @ e_t.T).squeeze(), k=64).indices
f_3d_agg = F_3D[topk_idx].mean(dim=0)   # R^512

# Fused embedding
fused = torch.cat([e_v, e_t, f_3d_agg], dim=-1)  # R^1536
```

The top-k spatial aggregation selects the 3D points most semantically relevant to the instruction (e.g., for "pick up the cup," it selects points near cup-like regions) and pools their features — giving the policy a spatially-grounded scene summary aligned to the current instruction.

---

## Stage 3 — Action Generation (MLP Policy)

The fused embedding is passed through an MLP that maps the 1536-dimensional joint representation to a distribution over discrete action tokens.

```python
import torch.nn as nn

class ActionPolicy(nn.Module):
    def __init__(self, input_dim=1536, hidden_dims=[512, 256], n_actions=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], n_actions),
        )

    def forward(self, fused_embedding):
        logits = self.net(fused_embedding)
        return logits  # pass to softmax for action probability distribution
```

**Training objective:** Cross-entropy loss over ground truth action token sequences collected from expert demonstrations in Isaac Sim:

```
L = -Σ_t log P(a_t* | e_v, e_t, F_3D)
```

where `a_t*` is the ground truth action token at step `t`.

**Zero-shot operation:** At inference time, no ground truth actions are available. The policy outputs the highest-probability action token given the current observation and instruction — generalizing to instructions not seen during training via CLIP's shared embedding space.

---

## Discrete Action Space

Actions are represented as discrete tokens rather than continuous joint angles or end-effector poses. This simplifies the learning problem (classification vs. regression) and aligns with the VLA paradigm used in models like RT-2 and OpenVLA.

| Token | Action |
|---|---|
| `MOVE_LEFT` | Base navigation left |
| `MOVE_RIGHT` | Base navigation right |
| `MOVE_FORWARD` | Base navigation forward |
| `MOVE_BACK` | Base navigation backward |
| `ARM_UP` | End-effector translate up |
| `ARM_DOWN` | End-effector translate down |
| `ARM_FORWARD` | End-effector translate forward |
| `ARM_BACK` | End-effector translate back |
| `GRIPPER_OPEN` | Open gripper |
| `GRIPPER_CLOSE` | Close gripper |
| `ROTATE_LEFT` | Base rotation left |
| `ROTATE_RIGHT` | Base rotation right |
| `STOP` | Terminate episode |
| ... | *(additional tokens for fine manipulation)* |

**Action sequence generation:** At each timestep, the policy outputs one token. The robot executes the token, the environment advances one step, a new observation is captured, and the policy runs again — closed-loop control at the token level.

---

## Isaac Sim Integration

NVIDIA Isaac Sim provides photorealistic RGB-D rendering, accurate physics simulation, and ROS2-compatible robot interfaces — enabling sim-to-real transfer research.

**Observation pipeline:**

```python
from omni.isaac.core import World
from omni.isaac.sensor import Camera

world = World()
camera = Camera(
    prim_path="/World/Robot/Camera",
    resolution=(640, 480),
    depth=True   # RGB-D output
)

def get_observation():
    rgb = camera.get_rgb()      # H×W×3 uint8
    depth = camera.get_depth()  # H×W float32 (meters)
    return rgb, depth
```

**Action execution:**

```python
def execute_action(robot, action_token):
    if action_token == "GRIPPER_CLOSE":
        robot.gripper.close()
    elif action_token == "ARM_FORWARD":
        robot.end_effector.apply_velocity([0.05, 0, 0])
    elif action_token == "MOVE_FORWARD":
        robot.base.apply_velocity([0.1, 0, 0])
    # ... dispatch table for all tokens
    world.step()
```

---

## Zero-Shot Generalization

The core claim of this work is zero-shot generalization to novel instructions. This is enabled by CLIP's shared embedding space:

**Seen instruction:** `"pick up the red cube"`
→ CLIP text embedding `e_t` close to RGB-D observations of pick-up motions near red objects

**Unseen instruction:** `"grasp the crimson block"`
→ CLIP text embedding for "crimson block" is semantically close to "red cube" in embedding space
→ Policy receives similar fused embedding → predicts similar action tokens
→ Generalizes without retraining

**What enables this:**
- CLIP's 400M-pair training creates smooth semantic neighborhoods in embedding space
- Synonyms, paraphrases, and semantically related descriptions map to nearby embeddings
- The MLP policy learns to map *regions* of embedding space to action tokens, not specific strings

**Current limitation:** The MLP policy can only generalize within the semantic neighborhoods learned during training. Instructions that require reasoning about spatial relationships not seen during training (e.g., "place the cup to the left of the bowl") require the attention/transformer decoder extension (see Future Work).

---

## Current Status

| Component | Status |
|---|---|
| CLIP visual + text encoding | ✅ Complete |
| 3D feature lifting (RGB-D → point cloud → CLIP field) | ✅ Complete |
| Top-k spatial aggregation | ✅ Complete |
| MLP policy head | ✅ Complete |
| Isaac Sim RGB-D observation pipeline | ✅ Complete |
| Discrete action token executor | ✅ Complete |
| Expert demonstration data collection | 🔄 In Progress |
| MLP policy training + evaluation | 🔄 In Progress |
| Zero-shot generalization benchmarking | 🔄 In Progress |
| Quantitative results | ⏳ Pending |

---

## Future Work

### Transformer Decoder Policy (Attention-Based Action Generation)

The MLP policy treats the fused embedding as a flat vector, losing the spatial structure of the 3D feature field. The planned extension replaces the MLP with a transformer decoder that attends directly over the per-point 3D feature field:

```
Query:  text embedding e_t                      (what to do)
Keys:   per-point 3D features F_3D ∈ R^{N×512} (where things are)
Values: per-point 3D features F_3D

Attention(e_t, F_3D, F_3D) → spatially-grounded action context
→ transformer decoder → action token sequence (autoregressive)
```

This enables the policy to reason about spatial relationships between objects — "to the left of," "on top of," "near the edge of" — by attending to the relevant 3D regions conditioned on the instruction.

### Planned Experiments
- Task success rate on seen vs. unseen instructions (zero-shot generalization gap)
- Ablation: 2D CLIP only vs. 2D + 3D feature lifting (quantify 3D contribution)
- Ablation: MLP policy vs. transformer decoder policy
- Comparison against RT-2 / OpenVLA on Isaac Sim tasks

---

## Installation

```bash
git clone https://github.com/sarthak-talwadkar/VLA.git
cd VLA
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch ≥ 1.13
- CLIP (`pip install git+https://github.com/openai/CLIP.git`)
- NVIDIA Isaac Sim 2023.1+
- Open3D (point cloud processing)
- NumPy, SciPy, Matplotlib

---

## Usage

### Collect expert demonstrations in Isaac Sim

```bash
python collect_demos.py \
    --env configs/isaac_env.yaml \
    --n-demos 500 \
    --output data/demos/
```

### Train MLP policy

```bash
python train.py \
    --demos data/demos/ \
    --clip-model ViT-B/32 \
    --hidden-dims 512 256 \
    --epochs 100 \
    --batch-size 32 \
    --save-dir checkpoints/
```

### Run zero-shot inference

```bash
python inference.py \
    --weights checkpoints/best.pt \
    --instruction "pick up the red cube and place it in the bin" \
    --env configs/isaac_env.yaml
```

---

## Project Structure

```
VLA/
├── models/
│   ├── clip_encoder.py       # CLIP visual + text encoding, L2 normalization
│   ├── scene_3d.py           # RGB-D → point cloud → 3D CLIP feature field
│   ├── fusion.py             # Top-k spatial aggregation + embedding concat
│   └── policy.py             # MLP policy head (action token prediction)
├── env/
│   ├── isaac_sim.py          # Isaac Sim observation + action execution interface
│   └── action_space.py       # Discrete action token definitions + dispatcher
├── data/
│   ├── collect_demos.py      # Expert demonstration collection in Isaac Sim
│   └── dataset.py            # Demonstration dataloader
├── train.py                  # Policy training entry point
├── inference.py              # Closed-loop zero-shot inference
├── evaluate.py               # Task success rate evaluation
├── requirements.txt
└── README.md
```

---

## References

- Radford, A. et al. *"Learning Transferable Visual Models From Natural Language Supervision."* ICML 2021. [[Paper]](https://arxiv.org/abs/2103.00020) *(CLIP)*
- Peng, S. et al. *"OpenScene: 3D Scene Understanding with Open Vocabularies."* CVPR 2023. [[Paper]](https://arxiv.org/abs/2211.15654)
- Brohan, A. et al. *"RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control."* CoRL 2023. [[Paper]](https://arxiv.org/abs/2307.15818)
- Kim, M. et al. *"OpenVLA: An Open-Source Vision-Language-Action Model."* 2024. [[Paper]](https://arxiv.org/abs/2406.09246)

---

## Author

**Sarthak Talwadkar**
MS Robotics, Northeastern University
[LinkedIn](https://linkedin.com/in/sarthak-talwadkar) · [GitHub](https://github.com/sarthak-talwadkar)
