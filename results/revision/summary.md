
### LaFAN1 → Unitree G1 (29 motions)

| Method | files | frames | E_g (mm) | E_mpbpe (mm) | E_dir (deg) | E_cmp (mm) | E_cmp_dyn (mm) | flip (%) | joint acc (rad/s²) | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| GMR IK (per-pair JSON) | 29 | 180806 | 82.8 | 91.6 | 11.5 | 168.3 | 94.3 | 1.4 | 18.7 | 0 |
| Chain LM (submitted, GMR mapping) | 29 | 180806 | 117.5 | 117.5 | 18.1 | 149.9 | 95.9 | 2.1 | 15.1 | 0 |
| Auto Chain LM (config-free) | 29 | 180806 | 48.2 | 53.9 | 6.4 | 165.1 | 96.7 | 0.0 | 13.2 | 0 |

### Per-motion distribution of E_dir_deg (deg)

| Method | Min | Median | Mean | Max |
|---|---|---|---|---|
| GMR IK | 9.1 | 11.5 | 11.5 | 13.5 |
| Chain LM (submitted) | 11.4 | 16.6 | 18.1 | 30.5 |
| Auto Chain LM | 5.2 | 5.9 | 6.4 | 11.0 |

### Per-motion distribution of E_g_mpbpe_mean (mm, own reference)

| Method | Min | Median | Mean | Max |
|---|---|---|---|---|
| GMR IK | 79.3 | 82.5 | 82.8 | 89.8 |
| Chain LM (submitted) | 98.4 | 108.9 | 117.5 | 171.8 |
| Auto Chain LM | 39.9 | 48.3 | 48.2 | 54.7 |

### Ablation (Auto Chain LM, G1)

| Method | files | frames | E_g (mm) | E_mpbpe (mm) | E_dir (deg) | E_cmp (mm) | E_cmp_dyn (mm) | flip (%) | joint acc (rad/s²) | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| full (default) | 29 | 180806 | 48.2 | 53.9 | 6.4 | 165.1 | 96.7 | 0.0 | 13.2 | 0 |
| root yaw only | 29 | 180806 | 74.4 | 84.4 | 6.7 | 171.1 | 98.4 | 0.1 | 13.7 | 0 |
| bend = cross (submitted form) | 29 | 180806 | 47.5 | 53.2 | 6.7 | 165.3 | 96.6 | 0.1 | 13.3 | 0 |
| no bend term | 29 | 180806 | 47.6 | 53.3 | 7.2 | 165.1 | 96.6 | 0.1 | 12.8 | 0 |
| w_bend=0.1 | 29 | 180806 | 47.6 | 53.3 | 6.9 | 165.0 | 96.5 | 0.0 | 13.0 | 0 |
| w_bend=1.0 | 29 | 180806 | 49.2 | 54.9 | 6.1 | 165.5 | 97.2 | 0.0 | 13.8 | 0 |
| w_bend=3.0 | 29 | 180806 | 52.2 | 57.2 | 9.0 | 170.1 | 101.0 | 0.0 | 14.7 | 0 |
| w_reg=0 | 29 | 180806 | 48.1 | 53.8 | 6.4 | 165.0 | 96.6 | 0.0 | 16.3 | 0 |
| w_reg=0.01 | 29 | 180806 | 48.1 | 53.8 | 6.4 | 165.0 | 96.6 | 0.0 | 15.4 | 0 |
| w_reg=0.2 | 29 | 180806 | 49.9 | 55.6 | 7.0 | 166.2 | 97.7 | 0.1 | 9.0 | 0 |

### LaFAN1 → other robots

| Robot | method | frames | E_g own (mm) | E_dir (deg) | E_cmp_dyn (mm) | flip (%) | errors |
|---|---|---|---|---|---|---|---|
| berkeley_humanoid_lite | Auto | 180806 | 128.4 | - | - | - | 0 |
| booster_k1 | Auto | 180806 | 69.9 | - | - | - | 0 |
| booster_t1_29dof | GMR IK | 180806 | 75.9 | 10.1 | 85.4 | 0.0 | 0 |
| booster_t1_29dof | Auto | 180806 | 67.0 | 6.0 | 73.5 | 0.0 | 0 |
| engineai_pm01 | GMR IK | 180806 | 66.2 | 10.2 | 80.5 | 0.4 | 0 |
| engineai_pm01 | Auto | 180806 | 80.2 | 15.2 | 91.8 | 0.2 | 0 |
| fourier_gr3 | Auto | 180806 | 55.3 | - | - | - | 0 |
| fourier_n1 | GMR IK | 180806 | 67.3 | 9.7 | 77.5 | 0.0 | 0 |
| fourier_n1 | Auto | 180806 | 69.8 | 6.7 | 66.4 | 0.0 | 0 |
| galaxea_r1pro | Auto | 180806 | 191.8 | - | - | - | 0 |
| hightorque_hi | Auto | 180806 | 81.0 | - | - | - | 0 |
| kuavo_s45 | Auto | 180806 | 129.0 | - | - | - | 0 |
| pal_talos | Auto | 180806 | 118.9 | - | 292.0 | 0.0 | 0 |
| pnd_adam_lite | Auto | 180806 | 96.1 | - | - | - | 0 |
| stanford_toddy | GMR IK | 180806 | 45.7 | 28.5 | 53.9 | 49.1 | 0 |
| stanford_toddy | Auto | 180806 | 45.4 | 25.5 | 61.5 | 10.3 | 0 |
| tienkung | Auto | 180806 | 87.3 | - | - | - | 0 |
| unitree_h1 | Auto | 180806 | 89.5 | - | - | - | 0 |
| unitree_h1_2 | Auto | 180806 | 91.8 | - | - | - | 0 |

### Robot → robot (source: G1 motion retargeted from LaFAN1)

| Target | frames | E_g own (mm) | self joint err (deg) | path diff vs direct (deg) |
|---|---|---|---|---|
| unitree_g1 | 58000 | 1.8 | 8.5 | 8.5 |
| unitree_h1 | 58000 | 85.9 | - | 5.0 |
| booster_t1_29dof | 58000 | 57.2 | - | 4.7 |
| fourier_n1 | 58000 | 47.3 | - | 5.4 |
| engineai_pm01 | 58000 | 66.5 | - | 9.5 |

### Throughput (single core, G1, walk1_subject1 first 2000 frames)

| Method | ms/frame | fps |
|---|---|---|
| GMR IK (daqp) | 1.46 | 685 |
| Auto Chain LM | 4.88 | 205 |
| Chain LM (submitted) | 60.51 | 17 |

### Matching consistency (16 robots × 77 LaFAN1 files)

See `matching_consistency.txt`: 15/16 robots produce one identical chain+body mapping on all 77 files
(left/right decided correctly on all). Berkeley Humanoid Lite: same chain matching, leg interior
interpolation varies (6 key points per leg).
