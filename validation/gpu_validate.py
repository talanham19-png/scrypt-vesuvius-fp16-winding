"""Turing (sm_75) validation of winding_model_9um: fp32 vs CUDA fp16 autocast vs (stock) bf16 autocast.
Inputs: slabs npz (rebuilt from published rays on the box) + HF ckpt. Prints JSON. MIT (Scrypt)."""
import sys, json, time, importlib.util, numpy as np, torch
here = sys.argv[1]; ck_path = sys.argv[2]; slabs = sys.argv[3]
s = importlib.util.spec_from_file_location("wm", here + "/winding_model.py"); wm = importlib.util.module_from_spec(s); s.loader.exec_module(wm)
def decode(phase, valid, anchor=192, em=8, trim=2):
    lo = anchor
    while lo > 0 and valid[lo-1]: lo -= 1
    hi = anchor + 1
    while hi < len(valid) and valid[hi]: hi += 1
    lo = max(lo, em); hi = min(hi, len(phase) - em)
    reg = phase - phase[anchor]; lower = np.floor(reg[lo:hi-1]); cnt = np.maximum((np.floor(reg[lo+1:hi]) - lower).astype(np.int32), 0)
    sl = np.flatnonzero(cnt); rep = cnt[sl].astype(np.int64); seg = np.repeat(sl + lo, rep)
    st = np.cumsum(rep) - rep; within = np.arange(int(rep.sum())) - np.repeat(st, rep)
    lev = np.repeat(lower[sl], rep) + within + 1; base = reg[seg]; step = np.maximum(reg[seg+1] - base, np.float32(1e-9))
    pos = seg.astype(np.float32) + np.clip((lev.astype(np.float32) - base) / step, 0, 1)
    return pos[trim:-trim], lev[trim:-trim].astype(np.int16)
import os
if os.environ.get("NO_CUDNN") == "1": torch.backends.cudnn.enabled = False

import torch.nn.functional as F
def chunked_forward(model, slab_image, slab_valid, Lchunk=96):
    """Same math as WindingModel.forward; decoder + full-res head run in ray-axis chunks
    (halo 2 for the two 3x3x3 decoder convs; head is 1x1). Bit-equivalent in fp32 (box check 9.5e-7)."""
    image = slab_image.float(); valid = slab_valid.float()
    mass = valid.sum(dim=(1, 2, 3)).clamp_min(1.0); mean = (image * valid).sum(dim=(1, 2, 3)) / mass
    centered = (image - mean[:, None, None, None]) * valid
    std = (centered.square().sum(dim=(1, 2, 3)) / mass).sqrt(); image = centered / (std[:, None, None, None] + 1e-6)
    feats = torch.stack([image, valid], dim=1); skips = []
    for st in model.stages: feats = st(feats); skips.append(feats)
    feats = model.trunk_in(feats)
    for b in model.stem_blocks: feats = b(feats)
    tok = feats.permute(0, 2, 3, 4, 1)
    for blk in model.attention_blocks: tok = blk(tok)
    feats = tok.permute(0, 4, 1, 2, 3); L = feats.shape[-1]; outs = []
    for s0 in range(0, L, Lchunk):
        e = min(L, s0 + Lchunk); a = max(0, s0 - 2); b = min(L, e + 2)
        up = F.interpolate(feats[..., a:b], scale_factor=(2, 2, 1), mode="trilinear", align_corners=False)
        dec = model.decoder(torch.cat([up, skips[-2][..., a:b]], dim=1))[..., s0 - a:s0 - a + (e - s0)]
        fine = F.interpolate(dec, size=tuple(skips[0].shape[2:4]) + (e - s0,), mode="trilinear", align_corners=False)
        outs.append(model._pixel_shuffle_output(model.full_resolution_head(torch.cat([fine, skips[0][..., s0:e]], dim=1)), 2))
    inc = F.softplus(torch.cat(outs, dim=-1).float()); return {"phase": inc.cumsum(-1)}
dev = torch.device("cuda:0"); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
m = wm.WindingModel(ck["config"]["model"]); m.load_state_dict(ck["model"]); m.to(dev).eval()
z = np.load(slabs); img = torch.from_numpy(z["img"]); val = torch.from_numpy(z["valid"])
out = {"gpu": torch.cuda.get_device_name(0), "cap": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__,
       "bf16_supported": torch.cuda.is_bf16_supported(), "bf16_native": (lambda: (torch.cuda.is_bf16_supported(including_emulation=False)))() if "including_emulation" in torch.cuda.is_bf16_supported.__code__.co_varnames else None, "cudnn_enabled": torch.backends.cudnn.enabled, "arms": {}}
def run(arm, bs):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); phases = []; t0 = None; times = []
    for i in range(0, len(img), bs):
        x = img[i:i+bs].to(dev); v = val[i:i+bs].to(dev); torch.cuda.synchronize(); t = time.time()
        with torch.inference_mode():
            fwd = (lambda a, b: chunked_forward(m, a, b, int(os.environ.get("LCHUNK", "96")))) if arm.endswith("c") else (lambda a, b: m(a, b))
            base = arm.rstrip("c")
            if base == "fp32": o = fwd(x, v.bool())
            else:
                with torch.autocast("cuda", dtype=torch.float16 if base == "fp16" else torch.bfloat16): o = fwd(x, v.bool())
        ph = o["phase"].float(); torch.cuda.synchronize(); times.append(time.time() - t); phases.append(ph.cpu().numpy())
    ph = np.concatenate(phases); res = {"batch": bs, "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "sec_per_slab_after_first": round(float(np.mean(times[1:]) / bs) if len(times) > 1 else times[0] / bs, 3),
        "first_call_s": round(times[0], 2), "nonfinite": int((~np.isfinite(ph)).sum()), "rays": []}
    for k in range(len(ph)):
        rt = z["ref_t"][z["ref_off"][k]:z["ref_off"][k+1]]; rl = z["ref_l"][z["ref_off"][k]:z["ref_off"][k+1]]
        p, l = decode(ph[k, 64, 64], z["valid"][k, 64, 64])
        if len(p) and len(rt):
            j = np.abs(rt[:, None] - p[None]).argmin(1); d = np.abs(rt - p[j])
            res["rays"].append({"ray": int(z["rays"][k]), "n": len(p), "n_ref": len(rt), "within_0.25": int((d <= .25).sum()),
                                "mean_abs_dt": round(float(d.mean()), 4), "max_abs_dt": round(float(d.max()), 4), "level_agree": float((rl == l[j]).mean())})
    return res, ph
P = {}
ARMS = [tuple([a.split(":")[0], int(a.split(":")[1])]) for a in (sys.argv[4] if len(sys.argv) > 4 else "fp32:1,fp16:1,fp16:2,bf16:1").split(",")]
for arm, bs in ARMS:
    try:
        r, ph = run(arm, bs); out["arms"][f"{arm}_b{bs}"] = r; P.setdefault(arm.rstrip("c"), ph)
    except Exception as e:
        out["arms"][f"{arm}_b{bs}"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    torch.cuda.empty_cache()
if "fp16" in P and "fp32" in P:
    reg = lambda p: p - p[:, 64:65, 64:65, 192:193]
    d = np.abs(reg(P["fp16"]) - reg(P["fp32"]))[z["valid"]]; out["fp16_vs_fp32_phase"] = {"mean": float(d.mean()), "p99": float(np.percentile(d, 99)), "max": float(d.max())}
print(json.dumps(out, indent=1))
