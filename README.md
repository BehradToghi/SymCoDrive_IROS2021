# SYMCODRIVE
## Cooperative Autonomous Vehicles that Sympathize with Human Drivers
### Authors: Behrad Toghi, Rodolfo Valiente, Dorsa Sadigh, Ramtin Pedarsani, Yaser P. Fallah

This work is submitted to IROS 2021, for more information please visit: https://symcodrive.toghi.net/


This repo is still under construction, we are going to upload detailed instructions and trained models soon.

Temporary instructions:
To run:
`python scripts/rl_agents_scripts/debugger.py`

You can add as many configs or options you want in this file, just chosse different names, e.g., `options_1`, `options_2`, etc.

This is a wrapper around the `experiment.py` file so we can easily run different options and configs for debugging purposes.

This implementation is built on top of the OpenAI Gym environment provided by:
```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```
