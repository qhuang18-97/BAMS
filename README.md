# BAMS

This repository contains reference implementation for BAMS, **Multi-agent Cooperative Games Using Belief Map Assisted
Training**.

## Requirements
* OpenAI Gym 0.9.6
* PyTorch 0.11.0 (CPU)
* [visdom](https://github.com/facebookresearch/visdom)
* Predator-Prey and Traffic Junction [Environments](https://github.com/apsdehal/ic3net-envs)


## Environment Installation

First, clone the repo and install ic3net-envs which contains implementation for Predator-Prey and Traffic-Junction

```
git clone https://github.com/IC3Net/IC3Net
cd IC3Net/ic3net-envs
python setup.py develop
```

**Optional**: If you want to run experiments on StarCraft, install the `gym-starcraft` package included in this package. Follow the instructions provided in README inside that packages.


Next, we need to install dependencies for IC3Net including PyTorch. For doing that run:

```
pip install -r requirements.txt
```

## Running

Once everything is installed, we can run the using these example commands

Note: We performed our experiments on `nthreads` set to 16, you can change it according to your machine, but the plots may vary.

Note: Use `OMP_NUM_THREADS=1` to limit the number of threads spawned

### Predator-Prey

- IC3Net on easy version

```
python main.py --env_name predator_prey --nagents 5 --nthreads 1 --num_epochs 4000 --hid_size 64 --detach_gap 10 --lrate 0.001 --dim 12 --max_steps 40 --ic3net --vision 1 --recurrent
```


For larger environment, change the following arguments:
- `nagents` to 10
- `max_steps` to 80
- `dim` to 20


## Contributors

- Qinwei Huang 
- Chen Luo
- Qinru Qiu

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