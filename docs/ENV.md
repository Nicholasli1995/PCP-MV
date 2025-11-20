## Dependency
You are suggested to create an environment that meets the following dependencies. Other versions of libraries may also work but **have not been tested**.

- Python >= 3.7.11
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv-full](https://github.com/open-mmlab/mmcv) = 1.5.2
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.24.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)
- [numba](https://pypi.org/project/numba/) = 0.56.4
- [pandas](https://pypi.org/project/pandas/) = 1.3.5
- [motmetrics](https://pypi.org/project/motmetrics/) = 1.1.3

## Compile and install
Compile and install the extension modules at your project directory ${PCP_MV_DIR} with:
```bash
python setup.py develop 
```

