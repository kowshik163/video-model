# Unified Shared Global Workspace (SGW) — High-performance Video Analysis Architecture

Below is a single, integrated graphical representation and full design spec for the SGW-based video understanding system you described. It includes a diagram (Mermaid), detailed component responsibilities, data interfaces, routing logic, compute-scheduling heuristics, training & loss design, evaluation metrics, and implementation notes to help this architecture outperform standard video-analysis models.

---

```mermaid
flowchart TD
  subgraph INPUT[Input Stream]
    A[Video Frames\n(+ timestamps, audio, meta)]
  end

  subgraph PREPROC[Preprocessing & Low-cost Estimation]
    A --> B1[BG / Camera Motion Estimation\n(Optical flow, sparse SLAM, depth priors)]
    A --> B2[Low-cost Object & Motion Detector\n(lightweight 2D CNN, OCM proposal)]
  end

  subgraph SGW[Shared Global Workspace]
    direction TB
    SGWNode[Heterogeneous Spatiotemporal Graph\n(nodes: objects, bg patches, forces;\nedges: spatial/causal/semantic)]
  end

  B1 -->|global flow, cam pose| SGWNode
  B2 -->|object proposals| SGWNode

  subgraph EXPERTS[Parallel Experts \n(Dense cross-attention bus)]
    direction LR
    V[Visual Expert\n(Video Transformer / 2D+flow)]
    P[Physics Expert\n(Differentiable Physics / Learned Simulator)]
    S[Spatial/Geometry Expert\n(Dense SLAM / 4D reconstructor)]
    C[Object-Centric Memory & Slot Module\n(OCM / tracking slots)]
  end

  SGWNode --> V
  SGWNode --> P
  SGWNode --> S
  SGWNode --> C

  V <--> P
  V <--> S
  P <--> S
  C <--> V
  C <--> P

  subgraph ROUTER[Supervisor / Router]
    R[Supervisor Agent\n(Reasoning Transformer / gating LLM)\n- Consistency loss calc\n- Routing decisions\n- Compute scheduler]
  end

  V --> R
  P --> R
  S --> R
  C --> R
  R --> SGWNode

  subgraph OPTIONAL3D[3D Path (on-demand)]
    R --> |needs_3d=True| D3[3D CNN (detailed shape & dynamics)\n(spawned sparsely)]
    D3 --> SGWNode
  end

  subgraph OUTPUT[Refined Outputs]
    SGWNode --> O1[Scene Graphs\n(object states, relations)]
    SGWNode --> O2[Predictions\n(short/long term trajectories)]
    SGWNode --> O3[Anomaly / Violation Reports\n(physics inconsistency)]
    SGWNode --> O4[Compressed World Memory\n(key frames & embeddings)]
  end
```

---

## High-level design goals

* **Single source of truth**: SGW (heterogeneous spatiotemporal graph) holds fused state and versioned history. Every expert reads and writes to it each tick.
* **Parallel experts + dense cross-attention**: Experts operate simultaneously and attend to each other's internal activations (not only final outputs) to share inductive biases.
* **Supervisor as a router and reasoner**: Lightweight reasoning model computes consistency metrics, chooses when to wake heavy compute (3D CNNs), and resolves conflicts by multi-step questioning of experts.
* **Adaptive compute**: Use scene dynamics and confidence/violation scores to throttle expensive modules (compute-on-demand). Static scenes mostly use memory recall.
* **Object-centric memory**: Slot-based OCM maintains continuity across occlusion and long temporal horizons.

---

## Component details & interfaces

### 1) Input layer

* **Data**: Frames (H,W,3), timestamps, optional audio, low-bandwidth metadata (GPS, IMU).\
* **Output**: Pipelined micro-batches (frames + delta-ts) for consistent time-step processing.

### 2) Preprocessing & low-cost estimators

* **BG / Camera Motion Estimation**: Lightweight optical flow (sparse), IMU fusion, coarse depth priors. Produces global flow fields, camera pose delta, confidence map.
* **Lightweight Object Detector & OCM proposals**: Tiny 2D CNN proposals (fast), initial masks, objectness score, feature embeddings.
* **Interface to SGW**: aligned bounding boxes/masks, pose priors, confidence and embedding vectors.

### 3) Shared Global Workspace (SGW)

* **Representation**: Heterogeneous graph with typed nodes and edges. Nodes: object-slot, background patch, force-field, camera node; Edges: spatial adjacency, contact, support, causal.
* **Temporal versioning**: Each time-step writes a new graph delta; memory compaction prunes old irrelevant nodes using learned saliency.
* **Read/write API**: transactional updates with conflict resolution triggered by Supervisor.

### 4) Parallel Experts (Council)

* **Visual Expert**: Video Transformer that ingests frame patches + optical flow. Produces per-object features, segmentation refinement, short-term motion embeddings.
* **Physics Expert**: Differentiable simulator that consumes object masses (inferred), friction, contacts from SGW. Outputs predicted next-state, violation scores, and uncertainty bands.
* **Spatial Expert**: Dense SLAM / 4D reconstructor that holds static geometry, predicts collision surfaces before visual evidence.
* **OCM / Tracker**: Slot-based object memory; when vision confidence drops, slots predict object continuation using physics priors.

**Cross-attention bus**: Experts exchange keys/values—Physics can attend to visual latent maps to estimate mass distribution; Visual attends to physics to bias motion search windows.

### 5) Supervisor / Router

* **Inputs**: draft outputs from experts, SGW snapshot, global metrics.
* **Responsibilities**:

  * Compute multi-expert consistency losses (visual vs. physics vs. spatial).
  * Decide routing: when to trigger 3D CNNs or heavy re-render steps.
  * Resolve conflicts using a short chain-of-reasoning (LLM-like decoder) that outputs corrections & textual rationale.
  * Schedule compute: if scene is static for N frames -> freeze heavy experts and rely on memory.
* **Decision heuristics**:

  * If Physics Violation Score > threshold and VisualConfidence < threshold -> spawn investigative rollout (re-run visual search + ask physics for alternate hypotheses).
  * If objects in close proximity with high relative velocity -> force 3D path for local region.

### 6) On-demand 3D CNN

* **Triggering**: Supervisor sets `needs_3d` for object(s) and region(s) of interest (ROI). Runs sparsely and with attention masks to limit compute.
* **Outputs**: refined meshes, depth-temporal priors, fine-grained dynamics for contact-rich events.

### 7) Output & Memory

* **Outputs**: structured scene graph, future trajectory predictions, anomaly reports, compressed episodic memory.
* **Downstream**: analytics, RL loops, control stacks, human-readable reports including the Supervisor’s rationale.

---

## Training objectives & loss design

Use multi-task, multi-modal training with modular losses. Key terms:

* **L_vis**: detection & segmentation loss (focal + dice) for Visual Expert.
* **L_phys**: physics prediction loss (state MSE over positions/velocities) + contact/resolution loss.
* **L_consistency**: cross-expert consistency loss — penalize mutual contradictions (e.g., visual says static but physics predicts acceleration with high confidence).
* **L_slot**: object permanence loss for OCM (temporal continuity, re-identification).
* **L_supervisor**: teacher-forced reasoning loss when human-labeled corrections are available, plus contrastive alignment between textual rationale and graph deltas.
* **Curriculum**: start with synthetic simulated data (perfect labels) to learn physics priors, then fine-tune on real-world videos with domain randomization.

Loss schedule: weighted sum with adaptive weights learned via uncertainty-weighting (Kendall et al.) so the network focuses on the dominant error mode.

---

## Key algorithms & heuristics for beating SOTA

1. **Feature-level cross-attention**: Not only outputs but mid-layer activations are fused, allowing physics to inform low-level visual filters and vice versa.
2. **Sparse 3D activation**: 3D CNNs are run only on ROIs flagged by combined visual+physics saliency, reducing compute by orders of magnitude.
3. **Supervisor-led reasoning rollouts**: the Supervisor issues short multi-step hypotheses and queries the experts in a simulated loop to resolve contradictions before committing to SGW updates.
4. **Learned pruning & memory consolidation**: compress past frames to key embeddings, remove low-salience nodes automatically.
5. **Uncertainty-aware fusion**: every expert provides uncertainty estimates; fusion weighs contributions inversely to uncertainty and flags low-certainty facts for Supervisor review.

---

## Scalability & compute considerations

* **Shard SGW** across GPUs/TPUs: partition by spatial regions or by object-slot groups for parallelism.
* **Asynchronous expert micro-steps**: experts run at different rates (visual: 8–16 fps heavy model, physics: continuous low-latency sim), synchronized by Supervisor checkpoints.
* **Gradient checkpointing & mixed precision** for memory savings.
* **Edge vs cloud**: push low-cost detectors and OCM tracking to edge; heavy 3D analysis and large physics rollouts to cloud.

---

## Metrics and evaluation

* **Per-object state error**: position & velocity RMSE over multiple horizons.
* **Collision accuracy**: precision/recall for predicted collision events.
* **Long-term prediction**: average displacement error at 1s, 3s, 5s horizons.
* **Violation detection**: true positive rate for physics violations and false alarm rate.
* **Compute-efficiency**: useful FLOPs per correctly predicted event; latency at target fps.

---

## Implementation notes / practical checklist

* Build modular SDK: expert containers with typed read/write SGW API.
* Use differentiable message-passing library for SGW (PyG / DGL variant with custom ops for temporal deltas).
* Supervisor: a compact Transformer (e.g., 6–12 layers) fine-tuned for reasoning over graph embeddings rather than raw text to keep latency low.
* Pretrain physics expert on simulator suites (Mujoco / Brax / custom differentiable sim) with domain randomization.
* Maintain strong telemetry: per-expert confidence, violation histograms — feed to continual learning and automated dataset curation.

