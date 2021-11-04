
# No-regret Online Learning over Riemannian Manifolds

This repository is the official implementation of [No-regret Online Learning over Riemannian Manifolds].

**Copyright © Xi Wang & Zhipeng Tu** <wangxi14@mails.ucas.ac.cn>

## Requirements

To install requirements:
scipy, mumpy and matplotlib:
```
python -m pip install --user numpy scipy matplotlib
```
Pymanopt:
```
pip install --user pymanopt
```

## Code structure

experiment:
- contains three experiments in the paper

lib:
- contains object functions and some operations
  
manifold:
- defines Class HyperbolicSpace
  
solver
- defines Class OnlineSolver and three Subclasses OnlineGradientDescent, OnlineBandit and OnlineZeroth
- defines Class OfflineSolver to solve the offline optimum with time horizon

## Instructions
1. Open the folder "experiment" to choose the experiment;
2. Open config.py to set parameters in the optimization;
3. Run genarate_A.py to generate data;
4. Run solve_offline.py to get offline optimum;
5. The solve_xxx.py to solve the problem in online sense;
6. Run plot.py to get figures