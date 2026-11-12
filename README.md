# DWM-DETR: Core Modules for Flame and Smoke Detection

This repository provides the three core modules of DWM-DETR: deformable feature extraction in the final backbone stage, dual-stream spatial and wavelet downsampling, and shared scale-conditioned mixture-of-experts encoding.

| Module | Source file | Implementation |
| --- | --- | --- |
| Deformable residual stage | [dcn_stage5.py](dcn_stage5.py) | Two residual blocks in the final R18-vd backbone stage, with four 3 x 3 modulated deformable convolutions and the corresponding residual shortcuts. |
| Dual-stream wavelet downsampling (DSWD) | [dswd.py](dswd.py) | A spatial convolution branch and a Haar branch using all four subbands. A learnable sigmoid-gated scalar weights the wavelet output before addition to the spatial output. |
| Shared scale-conditioned expert encoding (MoSME) | [mosme.py](mosme.py) | Shared self-attention, scale-conditioned token routing, an always-active shared expert, Top-2 selection from four routed experts, and output layer normalization. |

## Scope

The modules were extracted from the DWM-DETR implementation. Their core computations are retained, with the required activation helpers and two constructor functions included to illustrate module assembly.

This repository contains module-level source code. The complete detector, training framework, experiment configuration files, datasets, pretrained weights, and experimental logs are outside its scope. These files alone do not reproduce the full experiments reported in the manuscript.

## Requirements

- Compatible installations of PyTorch and torchvision.
- A working `torchvision.ops.deform_conv2d` operator for the deformable residual stage.
- `torch_scatter` is optional; the expert aggregation code retains a native PyTorch fallback.

## Usage

```python
import torch

from dcn_stage5 import build_deformable_stage5
from dswd import DualStreamDownsample
from mosme import build_shared_scale_encoder

stage5 = build_deformable_stage5(in_channels=256, out_channels=512)
downsample = DualStreamDownsample(in_channels=256, out_channels=256)
encoder = build_shared_scale_encoder()

# Example token sequences: [batch_size, number_of_tokens, hidden_dimension].
tokens_s4 = torch.randn(1, 64, 256)
tokens_s5 = torch.randn(1, 16, 256)

# Call the same encoder separately for the two scales to share its parameters.
# Positional embeddings may be supplied through the pos_embed argument.
encoder.eval()
with torch.no_grad():
    out_s4, balance_s4 = encoder(tokens_s4, scale_index=0)
    out_s5, balance_s5 = encoder(tokens_s5, scale_index=1)
```

DSWD expects an input feature map with even height and width. During training, the MoSME encoder returns an auxiliary balance-loss term alongside the encoded tokens. The calling training loop must incorporate that term into the objective if required by the training protocol; no training loop is included here.

## Verification

Python syntax and the abstract syntax trees of the extracted core definitions were checked against the supplied implementation. GPU training, inference, and operator execution were not tested as part of this extraction.

## Attribution

The implementation builds on [RT-DETR / RT-DETRv2](https://github.com/lyuwenyu/RT-DETR). Upstream copyright notices and license terms are retained in [NOTICE.md](NOTICE.md). Inclusion of the upstream license does not assign a new license to project-specific modifications.
