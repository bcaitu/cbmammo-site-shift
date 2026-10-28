
"""Analysis E: sensitivity of the opaque model's partial fine-tuning baseline and adaptation cost.
One process = one (encoder, set, seed). Draws = the first NDRAWS adaptation draws of analysis C (same make_draws seed rule), so the
head-refit variants of analysis C and the partial fine-tuning here use identical patients.
Configs vary one factor at a time around the manuscript's default (lr 1e-4, 6 epochs, last block, batch 8, weight decay 0.01)."""
import random
from rev1_common import *
from cbmammo.model import build, head_weights, masked_ce
from cbmammo.train_stage1 import class_weights as _cw
ENCN = os.environ["ENC"]; SET = os.environ["SET"]; SEED = int(os.environ["SEED"]); NDRAWS = int(os.environ.get("NDRAWS", 10)); K = int(os.environ.get("K", 25))
CFGS = [("default", 1e-4, 6, 1), ("lr3e-5", 3e-5, 6, 1), ("lr3e-4", 3e-4, 6, 1), ("ep3", 1e-4, 3, 1), ("ep12", 1e-4, 12, 1), ("blocks2", 1e-4, 6, 2)]
only = os.environ.get("CONFIGS"); CFGS = [c for c in CFGS if not only or c[0] in only.split(",")]
dev = torch.device("cuda")

def make_draws(y, pid, k, n, seed):
    rng = np.random.default_rng(seed); ups = np.unique(pid); out = []
    for _ in range(n):
        for _ in range(100):
            fp = rng.choice(ups, k, replace=False); inf = np.isin(pid, fp)
            fit, ev = np.flatnonzero(inf), np.flatnonzero(~inf)
            if len(np.unique(y[fit])) == 2 and len(np.unique(y[ev])) == 2: break
        out.append((fit, ev))
    return out

cbn, opqn = ENC[ENCN]; run_dir = f"{RUNS}/{opqn}_s{SEED}"
cfg = json.load(open(f"{run_dir}/run.json"))["config"]
d = dump(cbn, SEED, "test_" + SET); y, ok = target_arrays(d)
sid = np.array(d["sample_id"], dtype=object)[ok]; yy = (y[ok] == 1).astype(int); pid = patients(d)[ok]
R = {r["sample_id"]: r for r in D.load_manifest(["manifests/cmmd.jsonl", "manifests/cdd_cesm.jsonl"])}
draws = make_draws(yy, pid, K, NDRAWS, 7919 * SEED + 31 * K + len(SET))        # identical rule to analysis C
model = build(cfg).to(dev); model.load_state_dict(torch.load(f"{run_dir}/best.pt", map_location=dev), strict=False)
base = {k: v.clone() for k, v in model.state_dict().items()}
size = tuple(cfg["data"].get("size", [1024, 640])); hw = head_weights("opaque", cfg["model"].get("aux_lambda", 0.1), cfg["model"].get("target_weight", 1.0))
def loader(rs, train, bs):
    return torch.utils.data.DataLoader(D.BreastDataset(rs, size, train, cfg["data"].get("clahe", True), cfg["data"].get("cache_dir")), batch_size=min(bs, max(1, len(rs))),
                                       shuffle=train, num_workers=4, collate_fn=D.collate, drop_last=False)
rows = []
for dr, (fit, ev) in enumerate(draws):
    fr, er = [R[s] for s in sid[fit]], [R[s] for s in sid[ev]]
    el = loader(er, False, 16); cw = _cw(fr, dev)
    for name, lr_, epochs, blocks in CFGS:
        model.load_state_dict(base)
        for p in model.parameters(): p.requires_grad = False
        for blk in model.enc.m.blocks[-blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for mod in (model.fusion, model.pool, model.to_z, model.clf):
            for p in mod.parameters(): p.requires_grad = True
        model.enc.trainable = True
        params = [p for p in model.parameters() if p.requires_grad]; ntr = sum(p.numel() for p in params)
        torch.manual_seed(1000 * SEED + dr); random.seed(1000 * SEED + dr)
        opt = torch.optim.AdamW(params, lr=lr_, weight_decay=0.01)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); model.train()
        for _ in range(epochs):
            for b in loader(fr, True, 8):
                o = model(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev))
                loss, _ = masked_ce(o["logits"], b["labels"].to(dev), hw, cw)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        torch.cuda.synchronize(); t_fit = time.perf_counter() - t0; m_fit = torch.cuda.max_memory_allocated() / 2**30
        model.eval(); preds = []; t1 = time.perf_counter()
        with torch.no_grad():
            for b in el:
                o = model(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev)); preds.append(o["logits"]["malignant"].float().softmax(-1)[:, 1].cpu())
        torch.cuda.synchronize(); t_ev = time.perf_counter() - t1
        auc = fast_auc(yy[ev], torch.cat(preds).numpy())
        rows.append(dict(encoder=ENCN, set=SET, seed=SEED, draw=dr, config=name, lr=lr_, epochs=epochs, blocks=blocks, auc=auc, n_fit_breasts=len(fit), n_eval_breasts=len(ev),
                         trainable_params=ntr, t_fit_s=t_fit, t_eval_s=t_ev, peak_mem_fit_gib=m_fit, peak_mem_total_gib=torch.cuda.max_memory_allocated() / 2**30))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"E_pft_{ENCN}_{SET}_s{SEED}.csv"), index=False)
