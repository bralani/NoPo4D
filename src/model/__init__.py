from typing import Union

import torch.nn as nn

from .encoder import EncoderNoPo4DCfg
from .decoder import Decoder4DGSCfg
from .nopo4d import NoPo4D

MODELS = {
    "nopo4d": NoPo4D,
}

EncoderCfg = Union[EncoderNoPo4DCfg]
DecoderCfg = Decoder4DGSCfg


def get_model(encoder_cfg: EncoderCfg, decoder_cfg: DecoderCfg) -> nn.Module:
    return MODELS['nopo4d'](encoder_cfg, decoder_cfg)
