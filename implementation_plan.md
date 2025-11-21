# Implementation Plan: Unified Shared Global Workspace (SGW) Video Analysis Model

This document outlines the step-by-step plan to build a Python-based video analysis model that outperforms current state-of-the-art (SOTA) systems. The architecture is based on the **Shared Global Workspace (SGW)** theory, utilizing parallel experts, a reasoning supervisor, and adaptive compute.

## 1. Architecture Overview

The core innovation is the **SGW**, a dynamic spatiotemporal graph that serves as the "single source of truth." Unlike traditional pipelines that pass data sequentially, this system uses a "Council of Experts" that read/write to the SGW simultaneously, managed by a Supervisor.

### Key Differentiators (Why this wins):
*   **Physics Consistency**: Most models hallucinate motion. This model uses a Physics Expert to enforce valid trajectories.
*   **Adaptive Compute**: It doesn't run heavy 3D processing on every frame. The Supervisor routes compute only where needed (e.g., complex occlusions).
*   **Cross-Modal Attention**: Visual features inform physics parameters (mass/friction), and physics predictions guide visual search windows.

### Strategic Pivot (Addressing Engineering Risks):
*   **Avoid "Jack of All Trades"**: We will **freeze** the Physics Expert after pre-training on simulation. The Visual Expert must learn to map pixels to these fixed physical laws, preventing loss conflict.
*   **Solve "System ID"**: We rely on **Sim-to-Real** transfer. The Physics Expert is trained on Kubric (with ground truth mass/friction), and the Visual Expert is trained to infer these latent properties.
*   **Target OOD Generalization**: We compete on **Physion** and **VR-Bench**, not Kinetics-400. We aim for physical reasoning, not texture memorization.

## 2. Technology Stack

*   **Language**: Python 3.10+
*   **Deep Learning Framework**: PyTorch 2.x
*   **Graph Neural Networks**: PyTorch Geometric (PyG) or DGL
*   **Computer Vision**:
    *   *Detection*: YOLOv10 or RT-DETR (for speed/accuracy trade-off)
    *   *Optical Flow*: RAFT or FlowFormer
    *   *Backbone*: VideoMAE v2 or DINOv2 (ViT-based)
*   **Physics**: DiffTaichi or DeepMind's Graph Network Simulator (GNS)
*   **3D/Geometry**: PyTorch3D, Gaussian Splatting (for fast reconstruction)
*   **Orchestration**: Ray (for parallel expert execution)

## 3. Phase-by-Phase Implementation

### Phase 1: The Foundation (SGW & Data Structures)
**Goal**: Build the central data structure that holds the state of the world.

1.  **Define the Graph Schema**:
    *   **Nodes**: `ObjectNode` (ID, pos, vel, class, embedding), `BackgroundNode` (flow vector), `CameraNode` (pose).
    *   **Edges**: `SpatialEdge` (distance), `SemanticEdge` (interaction), `TemporalEdge` (past-to-future).
2.  **Implement the SGW Class**:
    *   Create a thread-safe class `SharedGlobalWorkspace` that manages the graph state.
    *   Implement `read_state(timestamp)` and `write_update(delta)` methods.
    *   Implement **Memory Pruning**: A mechanism to archive old nodes to long-term memory and keep the active graph lightweight.

### Phase 2: Input & Preprocessing Pipeline
**Goal**: Convert raw video into graph nodes efficiently.

1.  **Ingestion Module**:
    *   Build a `VideoLoader` using `decord` or `OpenCV` for fast frame access.
2.  **Low-Cost Estimators**:
    *   **Motion**: Integrate **RAFT** to compute dense optical flow. Downsample for the global flow field.
    *   **Detection**: Integrate **YOLOv10**. Map detections to `ObjectNodes`.
    *   **Camera**: Use simple homography estimation from optical flow to estimate camera motion (ego-motion).
3.  **Graph Initializer**:
    *   Write a function that takes detections + flow and populates the initial SGW state for $t=0$.

### Phase 3: The Council of Experts
**Goal**: Build the specialized modules that refine the graph.

1.  **Visual Expert (The Eye)**:
    *   Implement a **Video Transformer** (e.g., TimeSformer or VideoMAE).
    *   **Input**: Cropped patches of objects + local flow.
    *   **Output**: Refined embeddings, segmentation masks, and visual velocity vectors.
2.  **Physics Expert (The Simulator)**:
    *   Implement a **Differentiable Physics Engine** (GNN-based).
    *   **Training**: **Pre-train on Simulation ONLY** (Kubric/ThreeDWorld) with ground truth mass/friction. **Freeze weights** during real-world fine-tuning.
    *   **Input**: Object positions, velocities, and inferred properties (mass, friction).
    *   **Output**: Predicted next state ($t+1$) and a *Violation Score* (how physically impossible is the current visual observation?).
3.  **Object-Centric Memory (The Tracker)**:
    *   Implement **Slot Attention**.
    *   Maintain identity across occlusions. If an object disappears visually, the Memory module keeps the node alive based on Physics predictions.

### Phase 4: The Supervisor (Router & Reasoner)
**Goal**: The "Brain" that manages resources and resolves conflicts.

1.  **Consistency Engine**:
    *   Calculate `L_consistency`: The divergence between Visual observation and Physics prediction.
2.  **Routing Logic (The Switch)**:
    *   Implement a lightweight **Transformer Classifier**.
    *   **Gradient Strategy**: Use **Gumbel-Softmax** to allow differentiable "soft" routing during training, or **REINFORCE** (RL) if hard decisions are strictly required.
    *   **Logic**:
        *   If `Consistency < Threshold`: Trust the Visual Expert.
        *   If `Consistency > Threshold` (Conflict): Trigger the **3D Expert**.
        *   If `Scene_Static`: Sleep heavy experts, use Memory.
3.  **Conflict Resolution**:
    *   If Physics says "falling" but Vision says "hovering", the Supervisor checks for "support" edges in the graph. If none, it flags an anomaly or corrects the Visual Expert (maybe it's a poster on a wall, not a real object).

### Phase 5: The On-Demand 3D Expert
**Goal**: High-fidelity analysis for ambiguous situations.

1.  **3D CNN / Reconstruction**:
    *   Integrate a **3D CNN** (like X3D) or a fast reconstruction module (Gaussian Splatting).
    *   **Trigger**: Only runs when `Supervisor` sets `needs_3d=True` for a specific ROI.
    *   **Output**: Precise depth, 3D bounding boxes, and contact points.

### Phase 6: Training & Optimization (The "Pivot" Strategy)

1.  **Step 1: The Physics Prior (Simulation)**:
    *   Generate 1M+ clips using **Kubric** with varying mass, friction, and restitution.
    *   Train the **Physics Expert** to predict $\Delta State$ given ground truth parameters.
    *   **CRITICAL**: Freeze the Physics Expert weights. It is now the immutable "Law of Physics."

2.  **Step 2: System Identification (Visual Adaptation)**:
    *   Train the **Visual Expert** to predict the *inputs* for the Physics Expert (mass, friction, position) from pixels.
    *   Loss: $L_{sys\_id}$ (Difference between Visual-inferred parameters and Ground Truth in Sim).

3.  **Step 3: The Supervisor (Policy Learning)**:
    *   Train the **Supervisor** using RL (PPO) or Gumbel-Softmax.
    *   Reward: Accuracy on future prediction - Compute Cost.
    *   This teaches the router to only call the 3D Expert when the Physics/Visual experts disagree significantly.

4.  **Step 4: Real-World Fine-tuning**:
    *   Run on real video (Physion).
    *   Keep Physics Frozen.
    *   Fine-tune Visual Expert and Supervisor using Self-Supervised Consistency Loss ($L_{consistency}$).

3.  **Loss Functions**:
    *   Implement the custom losses defined in `readme.md` (`L_vis`, `L_phys`, `L_consistency`).

## 4. Execution Strategy

1.  **Prototype**: Build a synchronous version first (Frame -> Preproc -> SGW -> Experts -> Supervisor -> Update).
2.  **Parallelize**: Use **Ray** to put experts on different GPUs/threads. The SGW becomes an async store.
3.  **Optimize**: Use TensorRT for the YOLO and RAFT components to ensure real-time preprocessing.

## 5. Evaluation Metrics

**Do NOT compete on Kinetics-400** (Texture bias).

**Primary Benchmarks**:
*   **Physion / Physion++**: Measures physical reasoning (stability, prediction).
*   **VR-Bench**: Video Reasoning Benchmark.
*   **IntPhys**: Intuitive Physics benchmark.

**Success Metric**:
*   **OOD Generalization**: High performance on scenarios with unseen object shapes/materials.
*   **Sample Efficiency**: Ability to learn from fewer real-world examples due to the strong Physics Prior.
