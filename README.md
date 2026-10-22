# Winding-model inference on 8 GB Turing GPUs (fp16 / chunked decode)

Scrypt, October 2026. Code MIT. No scan data is included; all inputs are the public PHercParis4 data and the public `scrollprize/winding_model_9um` checkpoint.

## Summary
villa's `infer_winding_volume.py` hard-codes `torch.autocast("cuda", dtype=torch.bfloat16)`. Turing GPUs (RTX 20xx, sm_75) have no native bf16; on our RTX 2070, PyTorch reports bf16 as supported but not native (`bf16_native=False`), i.e. emulated. On this hardware **bf16 diverges from fp32** at the 0.25-voxel crossing tolerance, while **fp16 autocast tracks fp32** closely. Stock fp32 does not fit in 8 GB (12.73 GB peak).

This repo provides:
- `patch/villa_precision_decode_chunk.patch`: adds `--precision {auto,fp16,bf16,fp32}` (**default stays `bf16`**, so existing runs are unchanged; `auto` picks bf16 only with native support, else fp16) and `--decode-chunk-l N` (ray-axis chunked decoder/head, halo 2; fp32 max |Δ| 9.5e-7 vs unchunked). Base: villa `e2254ebe` (see `patch/BASE_COMMIT`).
- `tests/test_winding_precision.py`: 15 unit tests (CPU).
- `validation/gpu_validate.py`: the RTX 2070 comparison script.
- `results/`: per-ray results, ray selections and a ray-ID map into the published store.

The finding is scoped to Turing / RTX 2070. We have not measured how the published Paris4 `winding_inference` store was produced and make no claim about other hardware.

## Method
- Checkpoint: HF `scrollprize/winding_model_9um` `ckpt_final.pth` (sha256 `805a868c…ad8f02`), the one named in the published Paris4 `winding_inference` manifest.
- Rebuild the 128×128×384 input slabs for published rays from the public masked CT (level 2).
- Arms: fp32, fp16 autocast, bf16 autocast. Decode the center ray with villa's `decode_center_ray`; match each published crossing to the nearest decoded one (tolerance 0.25 and 1.0 vox).
- Two independent ray sets (A, B: 55 rays each, no overlap), stratified flat / steep / edge / void. Sets A/B use CPU emulation of CUDA autocast; a 6-ray subset was rerun on a real RTX 2070.

## Results (crossings within 0.25 vox of the published store)

| Arm | Set A (1,114) | Set B (1,188) | Combined (2,302) |
|---|---:|---:|---:|
| fp32 | 99.73% | 100.00% | 2299 (99.87%) |
| **fp16** | 99.73% | 99.66% | **2295 (99.70%)** |
| bf16 | 38.96% | 38.97% | 897 (38.97%) |

At 1.0 vox, bf16 matches 2276/2302 (98.87%): it carries a systematic offset of about 0.3 vox (≈2.5% of the ~12-vox crossing spacing), which is why it fails the 0.25 cutoff.

fp16 misses: on set A they are the same near-miss class as fp32's. On set B fp16 misses 4 crossings that fp32 gets, on rays 4550 (31/32) and 4143 (24/27). On ray 4550 the missed crossing is dropped entirely (the nearest decoded crossing is one winding level away), so fp16 level agreement is **not** 100% on that ray (31/32).

### Real RTX 2070 (torch 2.11.0+cu130, cuDNN off, 6 rays / 160 crossings)

| Arm | within 0.25 | Peak mem | s / slab |
|---|---:|---:|---:|
| fp32 (chunked) | 160/160 | 6.25 GB | 2.43 |
| fp16 | 156/160 | 7.22 GB | 1.03 |
| fp16 chunked | 156/160 | **3.43 GB** | **0.63** |
| bf16 | 54/160 (33.8%) | 7.22 GB | 2.10 |

The GPU run reproduces the CPU-emulated fp16 behaviour crossing-for-crossing on these rays (including the 4550 and 4143 misses). Stock fp32 unchunked peaks at 12.73 GB.

## Reproduce
```bash
git clone https://github.com/ScrollPrize/villa && cd villa
git checkout e2254ebe && git apply /path/to/patch/villa_precision_decode_chunk.patch
cp /path/to/tests/test_winding_precision.py vesuvius/tests/neural_tracing/
# checkpoint: https://huggingface.co/scrollprize/winding_model_9um (ckpt_final.pth)
WINDING_MODEL_CKPT=/path/ckpt_final.pth python -m pytest -q vesuvius/tests/neural_tracing/test_winding_precision.py
# inference on Turing:
python vesuvius/src/vesuvius/neural_tracing/winding_models/infer_winding_volume.py ... --precision auto --decode-chunk-l 96
```
GPU comparison: `python validation/gpu_validate.py <dir with villa winding_model.py> <ckpt_final.pth> <slabs.npz> "fp32c:1,fp16:1,fp16c:1,bf16:1"` (set `NO_CUDNN=1` to match our run). Slabs are rebuilt from the public CT for the rays in `results/ray_id_map.json` and are not redistributed here.

## Limitations
- Sets A/B are CPU emulation of autocast; real-GPU evidence is the 6-ray RTX 2070 run with cuDNN disabled.
- No full-volume re-inference; downstream spiral-fit impact of fp16 vs bf16 not measured.

## Ray IDs
Ray IDs index our fit-window subset; `results/ray_id_map.json` maps each to the published store (shard, shard_ray, global_ray, origin, step), e.g. 4550→global 83546, 4143→67310.
