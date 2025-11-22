# Phase 4: Evaluation & Refinement Report

## 1. Overview
Phase 4 focused on evaluating the SGW system on real-world video data (`video1.mp4`), identifying failure modes specific to "phonk-style" edits (fast cuts, heavy VFX), and implementing targeted refinements.

## 2. Key Implementations
- **Hierarchical Logging**: Implemented a structured logging system (`src/utils/logger.py`) that captures per-frame model states, decisions, and saves visual crops of tracked objects.
- **Transition Detection**: Added a visual transition detector (`src/utils/transition_detector.py`) to identify hard cuts and motion spikes, feeding this signal into the Supervisor's complexity score.
- **Audio Integration**: Added audio beat detection (`src/utils/audio_beats.py`) to further inform the Supervisor about temporal structure.
- **Optimization**: Implemented a fast optical flow path (Farneback) to enable rapid iteration.
- **Tooling**: Created `scripts/prepare_training_data.py` to harvest crops for retraining and `scripts/analyze_session.py` to visualize system behavior.

## 3. Failure Mode Analysis (Video 1)
- **Issue**: The initial run showed low 3D trigger rates (0.6%) despite the video being a dynamic "phonk edit".
- **Root Cause**: The Supervisor's "consistency score" (based on physics prediction error) was not sensitive enough to the *semantic* and *stylistic* shifts (cuts, wipes) typical of the genre. The object detector also likely struggled with partial occlusions and stylized clothing.
- **Mitigation**: 
    1.  **Explicit Signal**: We explicitly detect transitions and beats and force a higher complexity score, prompting the Supervisor to re-evaluate (triggering 3D expert or resetting tracks).
    2.  **Data Collection**: We are now harvesting crops of the specific "phonk" objects to fine-tune the detector.

## 4. Next Steps (Post-Phase 4)
1.  **Label Data**: Use the crops generated in `eval_logs/` to create a labeled dataset.
2.  **Fine-tune Visual Expert**: Retrain YOLOv5 on this new dataset to improve detection of stylized clothing/objects.
3.  **Train Policy**: Use the `analyze_session.py` insights to train the Supervisor's policy network (RL) to optimize the trade-off between 3D usage and accuracy, rather than using hard-coded heuristics.

## 5. Conclusion
The system infrastructure is now robust enough to handle complex video styles. The "loop" is closed: we can run inference, detect failures, harvest data, and prepare for retraining.
