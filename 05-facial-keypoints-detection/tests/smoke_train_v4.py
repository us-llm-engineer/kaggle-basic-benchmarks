"""End-to-end smoke test on CPU: train -> evaluate -> submission for each v4 config, on a tiny subset, 2-3 epochs.
Catches wiring errors (shapes, phases, decoders, submission format) before paid GPU time. Not a quality measurement."""
import os, sys, json, tempfile, shutil
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fkd_v4_common as C          # noqa: E402
import fkd_v4_configs as B  # noqa: E402
import torch                        # noqa: E402

df = pd.read_csv(os.path.join(ROOT, "data/training.csv"), nrows=96)
test_df = pd.read_csv(os.path.join(ROOT, "data/test.csv"), nrows=8)
data = dict(imgs=C.decode(df).astype(np.float32), y=df[C.KEYPOINT_COLS].values.astype(np.float32),
            idx_tr=np.arange(64), idx_va=np.arange(64, 96), test_imgs=C.decode(test_df).astype(np.float32),
            test_ids=test_df["ImageId"].values)

work = tempfile.mkdtemp()
os.makedirs(os.path.join(work, "data"))
look = pd.read_csv(os.path.join(ROOT, "data/IdLookupTable.csv"))
look[look.ImageId.isin(test_df.ImageId)].to_csv(os.path.join(work, "data/IdLookupTable.csv"), index=False)
os.chdir(work)
ok = True
NOTEBOOKS = {
    "v4a": "05_v4a_training_recipe.ipynb",
    "v4b": "06_v4b_adaptive_wing_loss.ipynb",
    "v4ctrl": "07_v4_seed_control.ipynb",
}
for run, spec in B.RUNS.items():
    try:
        if run in NOTEBOOKS:
            # Execute the notebook's own configuration cell so literal and syntax bugs surface here.
            nbcells = json.load(open(os.path.join(ROOT, NOTEBOOKS[run])))["cells"]
            cfg_src = next("".join(c["source"]) for c in nbcells if c["cell_type"] == "code" and "CFG = " in "".join(c["source"]))
            ns = {"C": C}
            exec(cfg_src, ns)
            cfg, model = ns["CFG"], ns["model"]
        else:
            # v4c is documented as future work, so its configuration is tested directly.
            cfg, model = spec["cfg"].copy(), eval(spec["model"], {"C": C})
        cfg.update({"batch": 16, "a_max": 1, "epochs": 3})
        if "b_phases" in cfg:
            cfg["b_phases"] = [{**p, "epochs": 1} for p in cfg["b_phases"]]
        model, hist, best = C.train(f"smoke_{run}", model, data, cfg)
        hfile = json.load(open(f"results/smoke_{run}_history.json"))
        assert len(hfile["val_rmse"]) == len(hist["epoch"]) and len(hfile["val_per_kp"][-1]) == 30
        res = C.evaluate_and_save(f"smoke_{run}", model, data, cfg, decoders=("softargmax", "argmax_shift", "dark"))
        n = C.write_submission(f"smoke_{run}", model, data, cfg, tta=True, decoder="dark")
        sub = pd.read_csv(f"submission_smoke_{run}.csv")
        fin = all(np.isfinite(v) for k, v in res.items() if k.startswith("rmse"))
        good = fin and len(sub) == n and sub.Location.notna().all() and os.path.exists(f"checkpoints/smoke_{run}.pt") \
            and os.path.getsize(f"logs/smoke_{run}.log") > 0
        print(("PASS" if good else "FAIL"), run, f"epochs={len(hist['epoch'])} stages={sorted(set(hist['stage']))} "
              f"rmse_soft={res['rmse_softargmax']:.2f} rmse_dark={res['rmse_dark']:.2f} sub_rows={len(sub)} beta={res['beta_final']}")
        ok &= bool(good)
    except Exception as e:
        ok = False
        print("FAIL", run, repr(e))
shutil.rmtree(work)
sys.exit(0 if ok else 1)
