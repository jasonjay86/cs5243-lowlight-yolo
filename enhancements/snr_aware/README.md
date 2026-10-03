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
### Vendoring the weights

Source: [JIA-Lab-research/SNR-Aware-Low-Light-Enhance](https://github.com/JIA-Lab-research/SNR-Aware-Low-Light-Enhance),
paper [PDF](https://jiaya.me/papers/cvpr22_xiaogang.pdf)
(Xu, Wang, Fu, Jia — CVPR 2022).

```bash
# After activating the cs5243-lowlight-yolo conda env:
pip install gdown     # one-time, also pinned in environment.yml
cd enhancements/snr_aware/weights
gdown 1g3NKmhz7WFLCm3t9qitqJqb_J7V4nzdb -O bundle.zip
# Bundle is 1.02 GB; contains all 7 SNR-Aware checkpoints under pretrain_model/.
# Inspect before extracting so you know exactly what's there:
python -c "import zipfile; [print(n) for n in zipfile.ZipFile('bundle.zip').namelist()]"
# We use LOLv1.pth (general low-light). Swap indoor_G.pth if your downstream
# data is very-dark indoor video; otherwise LOLv1 generalises better to
# outdoor camera-trap imagery.
python -c "import zipfile, shutil, os; \
  zipfile.ZipFile('bundle.zip').extract('pretrain_model/LOLv1.pth', '.'); \
  shutil.move('pretrain_model/LOLv1.pth', 'LOLv1.pth'); \
  shutil.rmtree('pretrain_model'); os.remove('bundle.zip')"
# Verify
python -c "import torch; sd = torch.load('LOLv1.pth', map_location='cpu', weights_only=True); \
  print('Loaded', len(sd), 'tensors'); print('First keys:', list(sd.keys())[:5])"
```

Expected smoke-test output: `Loaded 118 tensors` with first keys starting
`conv_first_1.weight`, `conv_first_2.weight`, `conv_first_3.weight`. The
three `conv_first_*` layers signal the paper's 3-branch SNR-aware
generator (one branch per SNR region); the rest of the state_dict maps
to down/up blocks and the SNR-aware transformer itself.

The `**/weights/` rule in the project root `.gitignore` covers the file
— `git check-ignore -v enhancements/snr_aware/weights/LOLv1.pth` should
print `.gitignore:27:**/weights/   ...`.

License: research-use, cite the paper (no explicit LICENSE file upstream):

```bibtex
@inproceedings{xu2022snr,
  title={SNR-aware Low-Light Image Enhancement},
  author={Xiaogang Xu and Ruixing Wang and Chi-Wing Fu and Jiaya Jia},
  booktitle={CVPR},
  year={2022}
}
```

## Implementation notes

- Input: uint8 BGR tensor. Output: uint8 BGR tensor (same shape).
- Match the eval harness contract: same shape, same dtype as input.
- Resolution: handle YOLO's expected input size (640×640 typically). The
  enhance function should be resize-agnostic — enhancement first, resize later.

## Latency reporting

SNR-Aware is **the expensive** method in the pipeline. Report:
- `flops_per_megapixel()` (transformer → expect single-digit to low-tens of GFLOPs/Mpx)
- `avg_latency_ms_per_image()` on the **reference GPU** the team agrees on.
