from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from mmdet.models import BACKBONES

class HighResolutionModule(nn.Module):
    def __init__(
        self,
        norm_cfg, 
        num_branches=4, 
        num_blocks=[1,1,2,4],
        num_inchannels=[32,64,128,128],
        num_channels=[32,64,128,128],
        fuse_layer_in_channel = [32, 64, 128, 128],
        fuse_layer_out_channel = [32, 64, 128, 128],        
        fuse_layer_kernel_size = [(1,1,6), (1,1,4), (1,1,3), (1,1,3)],
        fuse_layer_stride = [2, (1,1,2), (1,1,2), (1,1,2)],        
        multi_scale_output=True
        ):
        super(HighResolutionModule, self).__init__()
        self.norm_cfg = norm_cfg
        self.fuse_layer_in_channel = fuse_layer_in_channel
        self.fuse_layer_out_channel = fuse_layer_out_channel
        self.fuse_layer_kernel_size = fuse_layer_kernel_size
        self.fuse_layer_stride = fuse_layer_stride
        self._check_branches(
            num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, num_blocks, num_channels)
        self._make_fuse_layers()

    def _check_branches(self, num_branches, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, num_blocks, num_channels,
                         stride=1, expansion=1):
        layers = []

        sp_conv_layer = make_sparse_convmodule(
            self.num_inchannels[branch_index],
            num_channels[branch_index],
            kernel_size=3,
            stride=1,
            norm_cfg=self.norm_cfg,
            padding=1,
            indice_key="spconv_hr_b_{}_l_{}".format(branch_index, 0),
            conv_type="SparseConv3d"
        )

        layers.append(sp_conv_layer)
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * expansion
        
        for i in range(1, num_blocks[branch_index]):
            sp_conv_layer = make_sparse_convmodule(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                kernel_size=3,
                stride=1,
                norm_cfg=self.norm_cfg,
                padding=1,
                indice_key="spconv_hr_b_{}_l_{}".format(branch_index, i),
                conv_type="SparseConv3d"
            )
            layers.append(sp_conv_layer)            

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        self.fuse_layer1 = make_sparse_convmodule(
            self.fuse_layer_in_channel[0],
            self.fuse_layer_out_channel[0],
            kernel_size=self.fuse_layer_kernel_size[0],
            stride=self.fuse_layer_stride[0],
            norm_cfg=self.norm_cfg,
            padding=0,
            indice_key="hr_fuse_1",
            conv_type="SparseConv3d"            
        )
        self.fuse_layer2 = make_sparse_convmodule(
            self.fuse_layer_in_channel[1],
            self.fuse_layer_out_channel[1],
            kernel_size=self.fuse_layer_kernel_size[1],
            stride=self.fuse_layer_stride[1],
            norm_cfg=self.norm_cfg,
            padding=0,
            indice_key="hr_fuse_2",
            conv_type="SparseConv3d"            
        )   
        self.fuse_layer3 = make_sparse_convmodule(
            self.fuse_layer_in_channel[2],
            self.fuse_layer_out_channel[2],
            kernel_size=self.fuse_layer_kernel_size[2],
            stride=self.fuse_layer_stride[2],
            norm_cfg=self.norm_cfg,
            padding=0,
            indice_key="hr_fuse_3",
            conv_type="SparseConv3d"            
        )
        self.fuse_layer4 = make_sparse_convmodule(
            self.fuse_layer_in_channel[3],
            self.fuse_layer_out_channel[3],
            kernel_size=self.fuse_layer_kernel_size[3],
            stride=self.fuse_layer_stride[3],
            norm_cfg=self.norm_cfg,
            padding=0,
            indice_key="hr_fuse_4",
            conv_type="SparseConv3d"            
        )             
        return 

    def get_num_inchannels(self):
        return self.num_inchannels

    def transform_feats(self, spatial_features):
        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0,1,4,2,3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features
    
    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        else:
            x = [self.branches[i](x[i]) for i in range(len(x))]
        f1 = self.fuse_layer1(x[0]).dense()
        f2 = self.fuse_layer2(x[1]).dense()
        f3 = self.fuse_layer3(x[2]).dense()
        f4 = self.fuse_layer4(x[3]).dense()
        
        f1 = self.transform_feats(f1)
        f2 = self.transform_feats(f2)
        f3 = self.transform_feats(f3)
        f4 = self.transform_feats(f4)
        f3 = nn.functional.upsample(f3, scale_factor=2)
        f4 = nn.functional.upsample(f4, scale_factor=2)
        return f1 + f2 + f3 + f4
    
@BACKBONES.register_module()
class HRSparseEncoderV2(nn.Module):
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
            fuse_layer_in_channel = [32, 64, 128, 128],
            fuse_layer_out_channel = [32, 64, 128, 128],              
            fuse_layer_kernel_size = [(1,1,6), (1,1,4), (1,1,3), (1,1,3)],
            fuse_layer_stride = [2, (1,1,2), (1,1,2), (1,1,2)],                
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
        self.encoder_paddings = encoder_paddings
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
        self.high_resolution_module = HighResolutionModule(
            fuse_layer_in_channel=fuse_layer_in_channel,
            fuse_layer_out_channel=fuse_layer_out_channel,
            fuse_layer_kernel_size=fuse_layer_kernel_size,
            fuse_layer_stride=fuse_layer_stride,
            norm_cfg=norm_cfg
            )
    
    @auto_fp16(apply_to=("voxel_features",))
    def forward_(self, voxel_features, coors, batch_size, **kwargs):
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
        
        spatial_features = self.high_resolution_module(encode_features)
        return spatial_features    