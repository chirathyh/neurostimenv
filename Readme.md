<p align="center">
<img src="img/logo.png" alt="GluCoEnv" width="477"/>
</p>

---
<h1>NeuroStimEnv: A Reinforcement Learning Environment for Closed-Loop Transcranial Brain
Stimulation of Biological Neural Circuits</h1>

![license](https://img.shields.io/badge/License-MIT-yellow.svg)
[![python_sup](https://img.shields.io/badge/python-3.10.12-black.svg?)](https://www.python.org/downloads/release/python-31012/)

**NeuroStimEnv** is a project to develop Reinforcement Learning (RL)-based transcranial stimulation treatment targeting different psychiatric and neurological disorders. The framework allows integration of different neural circuits using the NEURON simulation engine and develop RL algorithms to modulate the target neural oscillations (brain waves).

<div align="center">
<img src="img/neurostimenv.png" alt="GluCoEnv" width="477"/>
</div>

Using the project
--

<h4>Installation</h4>

* Create and activate a Python3.10.12 virtual environment. <br>
* Clone the repository: <code>git clone git@github.com:chirathyh/neurostimenv.git</code>.<br>
* Go to the project folder (neurostimenv): <code>cd neurostimenv</code>. <br> 
* Install the required Python libraries <code>requirements.txt</code>. <br>
* Create an environment file <code>.env</code> at the root of the project folder with <code>MAIN_PATH=path-to-this-project</code> (<code>echo "MAIN_PATH=$(pwd)">.env</code>).<br>
* Compile and load NMODL (.mod) files, which define custom biophysical mechanisms like ion channels, synapses, and other membrane properties. Run command <code>nrnivmodl /path/to/modfiles</code> for the target neural circuit (e.g., <code>nrnivmodl /setup/circuits/L23Net/mod</code>).
---

<h4>Prerequsites</h4>

* The environment (i.e., biological neural circuit) is based on the gym framework. <br>
* The neural environment is simulated using <code>NEURON==8.2.3</code> and <code>LFPy==2.3</code>. <br>
* You can download different neural circuit models from ModelDB (https://modeldb.science/). <br>
* The project uses the <code>SimNIBS==4.1.0</code> for obtaining parameters for the transcranial stimulation. <br>

<h4>Quick Start - Running RL algorithms</h4>

Running a stochastic multi-arm bandit algorithm for a simple Ball and Stick Model.
```
cd experiments/bandit 
python mpirun -np 2 python run_mbandit.py experiment.name=test env=ballnstick agent=mbandit env.network.syn_activity=True experiment.tqdm=False agent.pretrain=True agent.checkpoint=test6
```

Running a Implicit Q-Learning algorithm for a depression mirocircuit (HL23Net: https://pubmed.ncbi.nlm.nih.gov/35021088/).
```
cd experiments/drl 
mpirun -np 2 python run_iql.py experiment.name=test env=hl23net env.network.syn_activity=True experiment.debug=True experiment.tqdm=False experiment.plot=False
```

<h4>Important Notes</h4>

* Running large neural circuits require a high-performance computing environment (e.g., <code>env=hl23net env.network.dt=0.025</code> with N=1,000 neurons and at dt=0.025). <br>
* To test the framework on your local machine, use the <code>env=ballnstick</code>, which uses a simple circuit (N=40) without any biophesical mechanisms. <br>
* Setting up neural circuit models
  - mod files: These are the mechanisms (e.g., Ca+2, K+ channels).
  - compile the files: <code>nrnivmodl</code>
  - model files
  - morphology files

### Citing
```
@misc{neurostimenv,
     author={NeuroStimEnv Team},
     title={NeuroStimEnv (2025)},
     year = {2025},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/chirathyh/neurostimenv}},
   }
```


Contact
--
Chirath Hettiarachchi - chirath.hettiarachchi@anu.edu.au\
School of Computing, Australian National University. 


TODO
--

Run times:
- N=100 neurons, setup 0.088 min (np=10, my laptop), runtime 1.435 min (4000ms at dt=0.512)
- N=1000 neurons, setup 1.735 min (np=16, ohiohgpu), runtime 21 min (4000ms at dt=0.512)
- N=1000 neurons, setup 27 mins, (np=16, nci), runtome 256 min (28,000 ms at 0.512)



