"""
A backward view-transform module.
"""

import torch
import torch.nn as nn

from torch.nn.functional import grid_sample

from mmdet3d.models.builder import VTRANSFORMS

def get_voxel_coord(bound1, bound2, voxel_size):
    start = bound1 + 0.5 * voxel_size
    n_voxels = int((bound2 - bound1) / voxel_size)
    end = bound2 - 0.5 * voxel_size
    coords = torch.linspace(start, end, n_voxels, dtype=torch.float32)
    return coords

@VTRANSFORMS.register_module()
class BackwardTransform(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=80,
        image_size=[720, 1920],
        xbound=[-96.0, 96.0, 0.2],
        ybound=[-48.0, 48.0, 0.2],
        zbound=[-10.0, 10.0, 0.2],
    ) -> None:
        super(BackwardTransform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.nx = [int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]]
        self._init_grid()
        self.proj_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 1)            
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),                        
        )        

    def _init_grid(self):
        xs = (
            get_voxel_coord(*self.xbound)
            .view(self.nx[0], 1, 1)
            .expand(-1, self.nx[1], self.nx[2])
            )
        ys = (
            get_voxel_coord(*self.ybound)
            .view(1, self.nx[1], 1)
            .expand(self.nx[0], -1, self.nx[2])
            )
        zs = (
            get_voxel_coord(*self.zbound)
            .view(1, 1, self.nx[2])
            .expand(self.nx[0], self.nx[1], -1)
            )
        grid = torch.stack((xs, ys, zs), -1)
        self.grid = nn.Parameter(grid, requires_grad=False)
        return                

    @force_fp32()
    def forward(
        self,
        img,
        lidar2image,
        **kwargs,
    ):
        batch_size = len(img)
        image_size = self.image_size
        nx, ny, nz, _ = self.grid.shape
        B, N, C, H, W = img.size()
        img = img.view(B*N, C, H, W)
        img = self.proj_net(img)
        BN, C, H, W = img.size()
        img = img.view(B, int(BN / B), C, H, W)

        bev_feats = []
        for b in range(batch_size):
            img_single = img[b]
            grid_coords = self.grid.view(-1, 3)

            cur_lidar2image = lidar2image[b]

            coords_proj = cur_lidar2image[:, :3, :3].matmul(grid_coords.T)
            coords_proj += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

            dist = coords_proj[:, 2, :]
            coords_proj[:, :2, :] /= coords_proj[:, 2:3, :]

            sample_grid = coords_proj.clone().permute(0, 2, 1)[:,:,None,:2]
            sample_grid[..., 0] = sample_grid[..., 0] / image_size[1] * 2 - 1
            sample_grid[..., 1] = sample_grid[..., 1] / image_size[0] * 2 - 1
            
            sampled_feats = grid_sample(img_single, sample_grid, align_corners=True)

            on_img = (
                (coords_proj[:, 0, :] < image_size[1]) 
                & (coords_proj[:, 0, :] > 0)
                & (coords_proj[:, 1, :] < image_size[0]) 
                & (coords_proj[:, 1, :] > 0)
                & (dist > 0.1)
            )

            sampled_feats = sampled_feats[..., 0].permute(0, 2, 1)
            sampled_feats[torch.logical_not(on_img)] = 0.
            sampled_feats = sampled_feats.view(N, nx, ny, nz, C)

            # pool across views
            sampled_feats = torch.max(sampled_feats, dim=0)[0]

            # average across the height dimension
            sampled_feats = sampled_feats.mean(dim=2).permute(2, 0, 1)
            bev_feats.append(sampled_feats[None, ...])

        bev_feats = torch.cat(bev_feats, dim=0)
        return bev_feats

