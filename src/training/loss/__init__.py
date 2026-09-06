from torch import nn
from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_opacity import LossOpacity, LossOpacityCfgWrapper
from .loss_depth_consis import LossDepthConsis, LossDepthConsisCfgWrapper
from .loss_life_span import LossLifeSpan, LossLifeSpanCfgWrapper
from .loss_ssim import LossSsim, LossSsimCfgWrapper
from .loss_optical_flow import LossOpticalFlow, LossOpticalFlowCfgWrapper
from .loss_distill import DistillLoss, LossDistillCfgWrapper
LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossOpacityCfgWrapper: LossOpacity,
    LossDepthConsisCfgWrapper: LossDepthConsis,
    LossLifeSpanCfgWrapper: LossLifeSpan,
    LossSsimCfgWrapper: LossSsim,
    LossOpticalFlowCfgWrapper: LossOpticalFlow,
    LossDistillCfgWrapper: DistillLoss,
}

LossCfgWrapper = (
    LossLpipsCfgWrapper
    | LossMseCfgWrapper
    | LossOpacityCfgWrapper
    | LossDepthConsisCfgWrapper
    | LossLifeSpanCfgWrapper
    | LossSsimCfgWrapper
    | LossOpticalFlowCfgWrapper
    | LossDistillCfgWrapper
)

def get_losses(cfgs: list[LossCfgWrapper]) -> list[nn.Module]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
