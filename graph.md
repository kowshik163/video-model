                                ┌───────────────────────────┐
                                │        INPUT VIDEO        │
                                │   (Frames + Timestamps)   │
                                └─────────────┬─────────────┘
                                              │
                            ┌─────────────────┼──────────────────┐
                            │                 │                  │
                            ▼                 ▼                  ▼
              ┌───────────────────┐ ┌───────────────────┐ ┌────────────────────┐
              │ 2. BG / Camera    │ │ 3-5. Object &     │ │ 7. Physics +        │
              │ Motion Estimation │ │ Motion Detector   │ │ Causal Reasoning    │
              │  (Optical Flow,   │ │  (2D CNN + OCM)   │ │  (Physics Engine +  │
              │   SLAM, Depth)    │ │                   │ │    Transformer)     │
              └───┬───────────────┘ └───────────┬───────┘ └───────────┬────────┘
                  │                             │                     │
                  │                             │                     │
                  ▼                             ▼                     ▼
         ┌────────────────┐           ┌───────────────────┐     ┌─────────────────────┐
         │ BG Motion Data │           │ Object Tracks     │    │ Reasoned World-State│
         │ (global flow,  │───────────│ (bbox, masks,     │────│ (predictions, causal│
         │ camera shift)  │           │ attributes, 2D vel)│   │ relations, physics) │
         └───────┬────────┘           └──────────┬─────────┘   └─────────┬───────────┘
                 │                               │                       │
                 │                               │                       │
                 │ ──────────────────────────────│───────────────────────│
                 │                               │                       │
                 │                 (Object Needs 3D?)                    │
                 │                         │                             │
                 │                         ▼                             │
                 │              ┌────────────────────┐                   │
                 │              │ ROUTER (Reasoning) │                   │
                 │              │ decides: send to   │                   │
                 │              │ 3D CNN or not      │                   │
                 │              └─────────┬──────────┘                   │
                 │                        │                              │
                 │                        │ YES                          │
                 │                        ▼                              │
                 │            ┌─────────────────────────┐                │
                 │            │ 9. 3D CNN (Compulsory)  │                │
                 │            │ (Detailed shape, 3D     │                │
                 │            │   dynamics, fine motion)│                │
                 │            └─────────────┬───────────┘                │
                 │                          │                            │
                 │                          ▼                            │
                 │             ┌─────────────────────────────┐           │
                 │             │ Refined 3D Motion + Objects │           │
                 │             └───────────────┬─────────────┘           │
                 │                              │                        │
                 │                              ▼                        │
                 │                   ┌──────────────────────┐            │
                 └──────────────────►│   CENTRAL REASONING   │◄──────────┘
                                     │  + WORLD STATE MEMORY │
                                     │  (Everything fused)   │
                                     └─────────────┬─────────┘
                                                   │
                                                   ▼
                                   ┌────────────────────────────────┐
                                   │ FINAL OUTPUTS                  │
                                   │ - Object motions               │
                                   │ - Future predictions           │
                                   │ - Physics-accurate trajectories│
                                   │ - Scene reconstruction         │
                                   └────────────────────────────────┘