# Stage 1-B Cable Tension Compensation + Z-Stiffness Boost (stage1_mp2)

## Baseline
- Branch parent: stage1_mp @ 2d66306 (feat(stage1): add Stage 1-B descent with quintic Hermite smoothing)
- Total score baseline: ~225

## Problem
Stage 1-B (7→3cm descent for SFP cable) consistently terminated with
```
axial err 13.6mm  (plug_z 0.1771m, target 0.1635m)
```
regardless of compensation magnitude. Root cause verified via diagnostic loop:
cable tension pulls plug upward. Hogan impedance steady-state:
`F = K·Δx  →  150 N/m · 0.0136 m = 2.04 N`

## Experiments

### Exp 1 — Feedforward compensation 14mm
- Commanded gripper z = target - 14mm
- Result: gripper physically stuck at z=0.2350m despite command change
- Score: 224.48 (Trial 1 75 full, Trial 2 50 partial, Trial 3 ~40)

### Exp 2 — Compensation 30mm (clamping vs equilibrium check)
- Commanded gripper z = target - 30mm
- Result: gripper STILL stuck at z=0.2350m (same as 14mm case)
- → Confirmed: equilibrium point, not clamping. Δx grows with K·Δx = F_cable(Δx).
- Score: 198.80 (−25.68). Larger compensation induced oscillation and worse Stage 2 handoff.

### Exp 3 — Option A: Z-stiffness boost 500 N/m (this commit)
- Stage 1-B convergence hold loop: K_z 150 → 500, D_z 70 → 130. XY/rot unchanged.
- SFP-only (detected via plug_name). S-curve body keeps low K for smoothness.
- Diagnostic shows plug_z static 0.1771m still (equilibrium unchanged),
  but boosted stiffness carries into Stage 2 transition, dominating cable disturbance.
- Score: **248.83** (+24.35 vs Exp 1, +50.03 vs Exp 2)
- Trial 2 SFP recovered full insertion (50 → 75). Trial 1 full retained. Trial 3 SC unchanged.

## Key Changes vs stage1_mp
- `Stage1Config.SFP_CABLE_TENSION_COMPENSATION` = 0.014 (feedforward only for SFP)
- `Stage1Config.STIFFNESS_MID_BOOST` / `DAMPING_MID_BOOST` — Z-only boost for hold loop
- `Stage1Config.STAGE1B_CONVERGENCE_TOL_M`, `STAGE1B_STABLE_CONSECUTIVE` — stability-based
  convergence (N consecutive err < tol) with per-0.3s diagnostic logging
- Convergence wait loop in Stage 1-B switches stiffness based on plug_name (SFP/SC)

## Remaining
- Trial 3 (SC) Stage 3 insertion failure — separate issue, unrelated to cable tension
- Reproducibility variance across seeds not yet measured (>=3 runs needed)
