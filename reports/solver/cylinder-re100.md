# Cylinder Re=100 acceptance

Overall: **PASS**

| Run | Grid (nx x ny) | Cd | St | Mass drift | Lift cycles | GPU s | Artifact SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| coarse | 384 x 480 | 1.547378 | 0.178530 | 2.705e-04 | 14.87 | 42.51 | `04333e92f5963bd42ba8cbd374b37b9bcb177d93fea448b2dda87140fff63a68` |
| canonical | 512 x 640 | 1.546266 | 0.178671 | 2.538e-04 | 11.16 | 95.41 | `0ac2a7df9a73030b2c0f202ab70e737fd355cfc7d177d5fb2f4a89a2f87253f4` |
| fine | 640 x 800 | 1.526614 | 0.178238 | 2.461e-04 | 8.91 | 130.56 | `9ecf3ae60f1eda1ba6fd7a08d776c6e26792dfb4fc3f9387ee6fffb018852624` |
| canonical repeat | 512 x 640 | 1.546266 | 0.178671 | 2.538e-04 | 11.16 | 95.41 | `0ac2a7df9a73030b2c0f202ab70e737fd355cfc7d177d5fb2f4a89a2f87253f4` |

- Source revision: `596172ce6cb5e78f8440699e648438bfa5bc0861`
- Lock SHA-256: `181b61f84e84aa57e5a373373de9b556c033d9510ceb98dfc78110a0e38bbb90`
- Canonical config SHA-256: `b15815f0d834795df582b986f6312425a9dc71601d04934375a58524a62a92d2`
- Device class: `L40S`
- Deterministic rerun: `True`
- Normalized three-grid study: `True`
- Cd adjacent changes contract: `False`
- Strouhal adjacent changes contract: `False`
- Report SHA-256: `b06833075668c12c15616cf6361f085314eb334973e79e2cd063a9e737d2b2a7`

The immutable reference intervals were not relaxed. Grid contraction booleans are
observations, not extra pass criteria; the sensitivity gate requires preserved normalized
geometry plus periodic, mass-stable runs on all three grids.
