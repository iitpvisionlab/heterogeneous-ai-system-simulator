# Predicting Performance of Heterogeneous AI Systems with Discrete-Event Simulations
Official source code for the "Predicting Performance of Heterogeneous AI Systems with Discrete-Event Simulations" paper.

## Authors
Vyacheslav Zhdanovskiy, Lev Teplyakov and Anton Grigoryev

## arXiv link
https://arxiv.org/abs/2204.03332

## Abstract
In recent years, artificial intelligence (AI) technologies have found industrial applications in various fields. AI systems typically possess complex software and heterogeneous CPU/GPU hardware architecture, making it difficult to answer basic questions considering performance evaluation and software optimization. Where is the bottleneck impeding the system? How does the performance scale with the workload? How the speed-up of a specific module would contribute to the whole system? Finding the answers to these questions through experiments on the real system could require a lot of computational, human, financial, and time resources. A solution to cut these costs is to use a fast and accurate simulation model preparatory to implementing anything in the real system. In this paper, we propose a discrete-event simulation model of a high-load heterogeneous AI system in the context of video analytics. Using the proposed model, we estimate: 1) the performance scalability with the increasing number of cameras; 2) the performance impact of integrating a new module; 3) the performance gain from optimizing a single module. We show that the performance estimation accuracy of the proposed model is higher than 90%. We also demonstrate, that the considered system possesses a counter-intuitive relationship between workload and performance, which nevertheless is correctly inferred by the proposed simulation model. 

## Requirements
```
numpy
simpy
```
See `requirements.txt` for more details.

## Launch instructions
```console
$ main.py --help
```

Example:
```console
$ ./main.py --graphs traces/*.graphml --graph-traces traces/*.traceml --nn-inferencer-traces traces/nn_inferencer --decoder-traces traces/decoder --video-streams 6
```

Example traces (collected on a real video analytics system) can be found [here](https://drive.google.com/file/d/1vzrij4XOSAMDpO_4aPRS_dy9ic065PZk/view?usp=sharing).

## Cite us
```
@article{zhdanovskiy2022predicting,
  title={Predicting Performance of Heterogeneous AI Systems with Discrete-Event Simulations},
  author={Zhdanovskiy, Vyacheslav and Teplyakov, Lev and Grigoryev, Anton},
  journal={arXiv preprint arXiv:2204.03332},
  year={2022}
}
```
