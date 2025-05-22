import torch

def distance_matrix(a, b):
    return torch.linalg.norm(a[:,None,:] - b[None, :, :], axis=2)

def get_density_weights(
        coords_map,
        centers,
        dist_thresh
        ):
    H, W, _ = coords_map.shape
    if len(centers) < 2:
        return torch.ones(H, W, dtype=torch.float32, device=centers.device)

    coors = coords_map.reshape(-1, 2)[:, None, :].to(centers.device)
    dist = torch.linalg.norm(coors - centers[None, :, :2], dim=2)
    flag = dist < dist_thresh
    weights = flag.sum(axis=1)
    weights[weights == 0] = 1.
    return weights.reshape(H, W)

def get_relationship_targets(
        bboxes, 
        max_dist=999., 
        dist_thresh=3
        ):
    xyzs = bboxes[:,:3]
    dm = distance_matrix(xyzs, xyzs)
    dm.fill_diagonal_(max_dist)

    targets = []
    for i in range(len(dm)):
        if dm[i].min() < dist_thresh:
            j = int(torch.argmin(dm[i]))
            targets.append(bboxes[j] - bboxes[i])
        else:
            targets.append(None)
    return targets
