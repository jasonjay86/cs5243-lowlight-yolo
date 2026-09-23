# SNR-Aware (Neural)

**Owner:** Jason Johnson

SNR-Aware is a **transformer-based low-light enhancement model** that uses
signal-to-noise ratio information to guide the enhancement. It is the
neural-network counterpart to CLAHE in our comparison.

Reference: Xu et al., "SNR-Aware Low-Light Image Enhancement" (2022).

## Files

- `__init__.py` — exports `SNRAwareEnhancement` implementing `Enhancement`.
- `model.py` — model definition (vendor the reference repo or depend via git submodule).
- `enhance.py` — inference wrapper (load weights → forward pass → uint8 output).
- `train.py` *(optional)* — only if we fine-tune on camera-trap data.
- `weights/` — pretrained weights. **Gitignored.** Place here manually or via download script.

## Setup

```bash
# Download pretrained weights (vendor link or reference repo)
mkdir -p weights
# curl / wget / manual download into weights/
```

Add the model checkpoint name to `.gitignore` if not already covered.

## Implementation notes

- Input: uint8 BGR tensor. Output: uint8 BGR tensor (same shape).
- Match the eval harness contract: same shape, same dtype as input.
- Resolution: handle YOLO's expected input size (640×640 typically). The
  enhance function should be resize-agnostic — enhancement first, resize later.

## Latency reporting

SNR-Aware is **the expensive** method in the pipeline. Report:
- `flops_per_megapixel()` (transformer → expect single-digit to low-tens of GFLOPs/Mpx)
- `avg_latency_ms_per_image()` on the **reference GPU** the team agrees on.
