import torch
from torch import nn

def power_average_pool(x, power):

    batch_size, channels, height, width = x.size()
    pooled = nn.functional.avg_pool2d(x, (height, width))
    pooled = torch.pow(pooled, power)
    pooled = torch.mean(pooled, dim=(2, 3), keepdim=True)
    return pooled

def CMDAF(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    prelu = nn.PReLU().cuda()

    sub_vi_ir = vi_feature - ir_feature
    vi_ir_div = sub_vi_ir * prelu(power_average_pool(sub_vi_ir, power=2))

    sub_ir_vi = ir_feature - vi_feature
    ir_vi_div = sub_ir_vi * prelu(power_average_pool(sub_ir_vi, power=2))


    # CAM
    vi_channel = vi_feature.mean(3).mean(2)  # shape: (batch_size, channels)
    ir_channel = ir_feature.mean(3).mean(2)  # shape: (batch_size, channels)
    vi_channel_weight = torch.unsqueeze(sigmoid(vi_channel), dim=2).unsqueeze(dim=3)
    ir_channel_weight = torch.unsqueeze(sigmoid(ir_channel), dim=2).unsqueeze(dim=3)
    vi_feature = vi_feature * vi_channel_weight
    ir_feature = ir_feature * ir_channel_weight

    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    # PAM
    batch_size, channels, height, width = vi_feature.size()

    vi_pos_weight = torch.sigmoid(torch.arange(height * width, device=vi_feature.device).view(1, 1, height, width))
    ir_pos_weight = torch.sigmoid(torch.arange(height * width, device=ir_feature.device).view(1, 1, height, width))

    vi_feature = vi_feature * vi_pos_weight
    ir_feature = ir_feature * ir_pos_weight

    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    # COA
    vi_corners = torch.stack([vi_feature[:, :, 0, 0], vi_feature[:, :, 0, -1],
                              vi_feature[:, :, -1, 0], vi_feature[:, :, -1, -1]], dim=2)
    ir_corners = torch.stack([ir_feature[:, :, 0, 0], ir_feature[:, :, 0, -1],
                              ir_feature[:, :, -1, 0], ir_feature[:, :, -1, -1]], dim=2)
    vi_corner_weight = torch.unsqueeze(sigmoid(vi_corners.mean(dim=2)), dim=2).unsqueeze(dim=3)
    ir_corner_weight = torch.unsqueeze(sigmoid(ir_corners.mean(dim=2)), dim=2).unsqueeze(dim=3)
    vi_feature = vi_feature * vi_corner_weight
    ir_feature = ir_feature * ir_corner_weight


    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature