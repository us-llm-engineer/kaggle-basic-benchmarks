"""Model and training configuration for each v4 run (shared by the training notebooks, tests and diagnostics)."""

AUG_FULL = {"affine": True, "rot_deg": 30, "scale": 0.20, "translate": 0.06, "photometric": True, "cutout": True}
AUG_FLIP_ONLY = {"affine": False, "photometric": False, "cutout": False}
BASE = {"epochs": 240, "a_max": 40, "a_patience": 6, "batch": 512, "lr": 1e-3, "lr_b": 1e-3, "sched": "step",
        "patience": 30, "seed": 42, "opt": "adamw", "wd": 1e-4}

RUNS = {
    "v4a": dict(
        title="v4a — training recipe: augmentation, temperature, joint supervision, flip test",
        model="C.SingleStageModel(H=24, n_deconv=1, beta=12.0)",
        cfg={**BASE, "H": 24, "sigma": 2.0, "heatmap_loss": "mse", "hm_frac": 0.2, "aug": AUG_FULL},
        why=("Same backbone and 24x24 head as v2; only the recipe changes. Integral Regression (Sun et al.) trains with "
             "translation +-2%, scale +-25%, rotation +-30 deg and flip, pre-trains the heatmap to saturation, finds joint "
             "heatmap + coordinate supervision (I1) best for 2D, and reports a flip-test gain. Adaptive Wing (Wang, "
             "Bo, Li Fuxin; Oregon State University) adds blur, noise and occlusion augmentation. Measured here before "
             "training: v2's softmax temperature (beta=1 on 0-1 Gaussian logits) puts soft-argmax ~14px off even an ideal "
             "heatmap; sigma=2 with beta=12 brings that floor to 0.03px, so the temperature is part of the recipe.")),
    "v4b": dict(
        title="v4b — Adaptive Wing Loss + Weighted Loss Map, 48x48 heatmap, CoordConv, DARK decoding (weighted)",
        model="C.SingleStageModel(H=48, n_deconv=2, coordconv=True, beta=15.0)",
        cfg={**BASE, "H": 48, "sigma": 2.0, "heatmap_loss": "awing", "hm_frac": 0.2, "aug": AUG_FULL},
        why=("Adaptive Wing (Oregon State University, arXiv 1904.07399) is the primary heatmap loss in this configuration. "
             "Its ablation: MSE 5.39 -> Adaptive Wing 4.65 -> + Weighted Loss Map 4.30 NME on WFLW, both free at inference "
             "; pose and expression subsets, the closest analogue to our nose-tip and lower-lip errors, improve 14-16% "
             ". The paper never tests coarse heatmaps or soft-argmax, so the heatmap is raised to 48x48 and "
             "CoordConv (+2 channels) is added as in the paper. DARK is evaluated at test time on the same "
             "heatmaps against soft-argmax and argmax + quarter shift.")),
    "v4c": dict(
        title="v4c — capacity: residual backbone + two-stage refinement head (MS-I1)",
        model="C.TwoStageModel(beta=12.0)",
        cfg={**BASE, "H": 24, "sigma": 2.0, "heatmap_loss": "mse", "hm_frac": 0.2, "aug": AUG_FULL},
        why=("Integral Regression's Table 3 (ResNet-18 -> 101: +4.6 PCKh@0.1) and Table 4 (a second stage: +2.7 "
             "PCKh@0.1, the largest single gain) say capacity and refinement matter. A ResNet-18-style backbone without "
             "the early max-pool (96x96 input), stage 1 on 12x12 features, stage 2 refining on 24x24 features concatenated "
             "with stage-1 heatmaps, both stages supervised; same recipe as v4a so the difference isolates capacity.")),
    "v4ctrl": dict(
        title="v4ctrl — seed-noise control: exact v2 configuration, seed 7",
        model='C.SingleStageModel(H=24, n_deconv=1, beta=1.0, learn_beta=False, mapping="v2")',
        cfg={"epochs": 0, "a_max": 8, "a_patience": 99, "batch": 256, "lr": 1e-3, "seed": 7, "opt": "adam",
             "H": 24, "sigma": 1.0, "mapping": "v2", "heatmap_loss": "mse", "hm_frac": 0, "aug": AUG_FLIP_ONLY,
             "b_phases": [{"epochs": 20, "lr": 5e-4, "sched": "plateau", "plateau_patience": 3, "patience": 5},
                          {"epochs": 250, "lr": 3e-4, "sched": "plateau", "plateau_patience": 6, "patience": 30}]},
        why=("The one re-run of the v2 configuration: v2's architecture, targets (sigma=1, v2 "
             "cell mapping), fixed beta=1, flip-only augmentation, Adam, batch 256, Stage A 8 epochs, then v2's two "
             "Stage B phases. Only the seed differs (7 vs 42). |v2 - control| is the noise band every v4 comparison is "
             "judged against.")),
}
