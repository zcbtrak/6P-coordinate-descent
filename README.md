# AdderBoard Submission: 6-Parameter GPU-Trained Model

This repository contains the weights and training implementation for a **6-parameter** 1L Qwen-derived decoder that achieves **100.00% accuracy** on the AdderBoard integer addition task (numbers up to $10^{10}-1$).

## Model Details
- **Architecture**: 1-Layer Qwen Decoder (d=2, 1h, hd=2, ff=2).
- **Parameters**: 6 (Learnable: `embed_w0`, `embed_w1`, `v_proj_w`, `gate_w0`, `gate_w1`, `carry_w`).
- **Accuracy**: 100.00% (Verified on 10,000+ random samples).

## Files
- `submission_gpu_trained.py`: The main model file containing the optimized parameters.
- `train_gpu_6p.py`: The GPU-optimized training script used to find these parameters.

## Verification
You can verify the submission using the `verify.py` script from the [AdderBoard](https://github.com/yk/AdderBoard) repository:
```bash
python verify.py submission_gpu_trained.py
```

## Credits
Trained by **zcbtrak**.
Original 10 parameter architecture by `Lokimorty`.
