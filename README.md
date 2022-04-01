# Go Explore

**Under development**

[![CI](https://github.com/qgallouedec/go-explore/actions/workflows/ci.yml/badge.svg)](https://github.com/qgallouedec/go-explore/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/qgallouedec/go-explore/branch/main/graph/badge.svg?token=f0yjhgL1nj)](https://codecov.io/gh/qgallouedec/go-explore)

Unofficial implementation of the Go-Explore algorithm presented in [First return then explore](https://arxiv.org/abs/2004.12919) based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)


## Installation

```bash
git clone https://github.com/qgallouedec/go-explore
cd go-explore
pip install -e .
```


## Usage


```python
from go_explore import GoExplore

go_explore = GoExplore("MontezumaRevenge-v0")
go_explore.explore(total_timesteps=100000)
go_explore.robustify(total_timesteps=1000)
```


## Disclaimer

Some components of the algorithm have not been implemented yet:

- (To be filled)

Contributions are welcome.
