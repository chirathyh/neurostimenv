<p align="center">
<img src="img/logo.png" alt="GluCoEnv" width="477"/>
</p>

<div align="center">

---

<h1>NeuroStimEnv: A Reinforcement Learning Environment for Stimulating Biological Neural Circuits</h1>

![license](https://img.shields.io/badge/License-MIT-yellow.svg)
[![python_sup](https://img.shields.io/badge/python-3.10.12-black.svg?)](https://www.python.org/downloads/release/python-31012/)
</div>

---

Setup
- Create a Python 3.10.12 virtual environment.
- Clone the repository
- Go to the project folder<code>cd NeuroStimEnv</code>
- Create a .env file.
- Run command <code>nrnivmodl</code> inside the /mod folder.


Features
- The environment is based on the gym framework.
- The neural environment is based on the NEURON 8.2.3 and LFPy 2.3.
- You can download different neural circuit models from ModelDB.

RL Information
- Observation/Sate: Measured EEG
- Action: tACS, todo: can extend to TMS

Setting up neural circuit models
- mod files: These are the mechanisms (e.g., Ca+2, K+ channels).
- compile the files: <code>nrnivmodl</code>
- model files
- morphology files

Run times:
- N=100 neurons, setup 0.088 min (np=10, my laptop), runtime 1.435 min (4000ms at dt=0.512)
- N=1000 neurons, setup 1.735 min (np=16, ohiohgpu), runtime 21 min (4000ms at dt=0.512)
- N=1000 neurons, setup 27 mins, (np=16, nci), runtome 256 min (28,000 ms at 0.512)


Install (prepare a requirements.txt file)
- SimNIBS==4.1.0
- NEURON==8.2.3
- LFPy==2.3
- python-decouple==3.4
- mpi4py==3.1.5
- gym==0.9.4
- pandas==2.2.0
- xlrd==2.0.1
- numpy==1.26.3

There is a conflict for scipy and numpy, fix
