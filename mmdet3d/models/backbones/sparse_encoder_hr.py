from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from mmdet.models import BACKBONES

@BACKBONES.register_module()
class HRSparseEncoder(nn.Module):
    def __init__(
            self, 
            in_channels, 
            sparse_shape,
            order=("conv", "norm", "act"),
            norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
            base_channels=16,
            output_channels=128,
            encoder_channels=((16,), (32,32,32), (64,64,64), (64,64,64)),
            encoder_paddings=((1,), (1,1,1), (1,1,1), ((0,1,1), 1,1)),
            block_type="conv_module"
            ):
        super().__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False

        assert isinstance(order, (list, tuple)) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        if self.order[0] != "conv":
            self.conv_input = make_sparse_convmodule(
                in_channels, 
                self.base_channels, 
                3, 
                norm_cfg=norm_cfg, 
                padding=1, 
                indice_key="subm1", 
                conv_type="SubMConv3d", 
                order=("conv",)
                )
        else:
            self.conv_input = make_sparse_convmodule(
                in_channels, 
                self.base_channels, 
                3, 
                norm_cfg=norm_cfg, 
                padding=1, 
                indice_key="subm1", 
                conv_type="SubMConv3d", 
                )            
        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, 
            norm_cfg, 
            self.base_channels, 
            block_type=block_type
            )
        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1,1,3),
            stride=1,
            norm_cfg=norm_cfg,
            padding=0, 
            indice_key="spconv_down2", 
            conv_type="SparseConv3d", 
            )          
        self.hr_proj_layer = make_sparse_convmodule(
            64,
            self.output_channels,
            kernel_size=(1,1,8),
            stride=(1,1,2), 
            norm_cfg=norm_cfg, 
            padding=0, 
            indice_key="hr_proj", 
            conv_type="SparseConv3d", 
            )  
    
    @auto_fp16(apply_to=("voxel_features",))
    def forward(self, voxel_features, coors, batch_size, **kwargs):
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)
        
        last_layer_feature = self.conv_out(encode_features[-1]).dense()
        upsampled_spatial_features = nn.functional.upsample(last_layer_feature, scale_factor=2)
        hr_branch_features = self.hr_proj_layer(encode_features[1]).dense()
        spatial_features = hr_branch_features + upsampled_spatial_features

        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features

    def make_encoder_layers(
        self,
        make_block,
        norm_cfg,
        in_channels,
        block_type="conv_module",
        conv_cfg=dict(type="SubMConv3d"),
    ):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ["conv_module", "basicblock"]
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="SparseConv3d",
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f"spconv{i + 1}",
                                conv_type="SparseConv3d",
                            )
                        )
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_type="SubMConv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = spconv.SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels