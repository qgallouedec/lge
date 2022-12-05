# Latent Go-Explore


[![codecov](https://codecov.io/gh/qgallouedec/go-explore/branch/main/graph/badge.svg?token=f0yjhgL1nj)](https://codecov.io/gh/qgallouedec/go-explore)

Official implementation of Latent Go-Explore (LGE) algorithm.

Paper: [Cell-Free Latent Go-Explore](https://arxiv.org/abs/2208.14928)


## Installation

```bash
git clone https://github.com/qgallouedec/lge
cd lge
pip install -e .
```

## Usage


```python
from stable_baselines3 import SAC

from lge import LatentGoExplore

lge = LatentGoExplore(SAC, "MountainCarContinuous-v0")
lge.explore(total_timesteps=10_000)
```

Supported envrionments specifications:

| Space           | Observation space  |
| --------------- | ------------------ |
| `Discrete`      | :heavy_check_mark: |
| `Box`           | :heavy_check_mark: |
| Image           | :heavy_check_mark: |
| `MultiDiscrete` | :x:                |
| `MultiBinary`   | :heavy_check_mark: |

| Space      | Action space       |
| ---------- | ------------------ |
| `Box`      | :heavy_check_mark: |
| `Discrete` | :heavy_check_mark: |



 Image is actually a multi-dimensonal uint8 `Box`.
