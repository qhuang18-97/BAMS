# BAMS

This repository contains reference implementation for BAMS, **Multi-agent Cooperative Games Using Belief Map Assisted
Training**.

## Cite

If you use this code or BAMS in your work, please cite the following:
```
@article{,
  title={Multi-agent Cooperative Games Using Belief Map Assisted Training},
  author={Luo, Chen and Huang, Qinwei and Wu, Alex and Khan, Simon and Li, Hai And Qiu, Qinru},
  journal={},
  year={2023}
}
```

## Requirements
* OpenAI Gym 0.9.6
* PyTorch 0.11.0 (CPU)
* [visdom](https://github.com/facebookresearch/visdom)
* Predator-Prey [Environments](https://github.com/apsdehal/ic3net-envs)


## Environment Installation

First, clone the repo and install ic3net-envs which contains implementation for Predator-Prey and Traffic-Junction

```
git clone https://github.com/qhuang18-97/BAMS.git
cd BAMS/ic3net-envs
python setup.py develop
```


Next, we need to install dependencies for IC3Net including PyTorch. For doing that run:

```
pip install -r requirements.txt
```

## Running

Once everything is installed, we can run the using these example commands


### Predator-Prey

- IC3Net on easy version

```
python main.py --env_name predator_prey --nagents 5 --nthreads 1 --num_epochs 4000 --hid_size 64 --detach_gap 10 --lrate 0.001 --dim 12 --max_steps 40 --ic3net --vision 1 --recurrent
```


For larger environment, change the following arguments:
- `nagents` to 10
- `max_steps` to 80
- `dim` to 20


## Reference

The training framework is adapted from [IC3Net](https://github.com/IC3Net/IC3Net)
```
@article{singh2018learning,
  title={Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks},
  author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
  journal={arXiv preprint arXiv:1812.09755},
  year={2018}
}
```
