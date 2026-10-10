"""Collect the char-TF-IDF MLP fold outputs (one method per n-gram setting N = 3, 4, 5, 25 = concatenated 2..5), temperature-scale each (T fitted on the other folds' OOF rows), save methods mlp_N{N}, and print training-dynamics summaries from the per-epoch histories."""
import json, os, sys
import numpy as np
import spooky_common as S
RUNS = os.path.join(S.RESULTS, "char_mlp_runs"); seed = sys.argv[1] if len(sys.argv) > 1 else "7"; train, test, folds, y = S.load()
for N in (3, 4, 5, 25):
    oof, tst, H, nf = np.zeros((len(y), 3)), 0, [], []
    for k in range(5):
        z = np.load(os.path.join(RUNS, f"N{N}_s{seed}_f{k}.npz")); oof[z["va"]] = z["pv"]; tst = tst + z["pt"] / 5; h = json.load(open(os.path.join(RUNS, f"N{N}_s{seed}_f{k}_history.json"))); H.append(h["history"]); nf.append(h["n_features"])
    cal, Ts, tcal = S.cv_temperature(oof, y, folds, tst); e, raw = S.evaluate(y, cal, folds), S.evaluate(y, oof, folds, 10)
    be = [int(np.argmin([r["inner_loss"] for r in h])) for h in H]; S.save_method(f"mlp_N{N}", cal, tcal, dict(method=f"shallow ANN (100,50) ReLU on TF-IDF char n-grams N={N} (arXiv 2506.15650 adaptation), from scratch", best_epochs=be, temperatures=Ts, n_features=nf, **e))
    print(f"mlp_N{N:<3d}: raw {raw['log_loss']:.4f} -> temperature-scaled {e['log_loss']:.4f} {np.round(e['log_loss_ci95'],4)} acc {e['accuracy']:.4f} | best epochs {be} | stopped at epochs {[len(h) for h in H]} | features {nf[0]} | T {np.round(Ts,2).tolist()}")
    b = [h[i] for h, i in zip(H, be)]; print(f"        at best epoch (mean over folds): train {np.mean([r['train_loss'] for r in b]):.3f} inner {np.mean([r['inner_loss'] for r in b]):.3f} conf {np.mean([r['inner_mean_conf'] for r in b]):.3f} ece {np.mean([r['inner_ece'] for r in b]):.3f}; final epoch: train {np.mean([h[-1]['train_loss'] for h in H]):.4f} inner {np.mean([h[-1]['inner_loss'] for h in H]):.3f} conf {np.mean([h[-1]['inner_mean_conf'] for h in H]):.3f} ece {np.mean([h[-1]['inner_ece'] for h in H]):.3f}")
