from neuron import h, gui

# Create a single neuron
soma = h.Section(name='soma')
soma.L = 20  # Length in µm
soma.diam = 20  # Diameter in µm
soma.insert('hh')  # Insert Hodgkin-Huxley mechanism

# Insert extracellular mechanism
for sec in h.allsec():
    sec.insert('extracellular')

# Attach extracellular stimulation
stim_vec = h.Vector()  # Define the extracellular potential as a Vector
stim_time = h.Vector()  # Time vector for the extracellular stimulation
stim_time.record(h._ref_t)  # Record the simulation time

# Define a sinusoidal extracellular potential (e.g., 100 Hz sine wave)
import numpy as np

stim_vec.from_python(0.1 * np.sin(2 * np.pi * 100 * np.linspace(0, 1, 10000)))

# Set up the extracellular stimulation to affect all sections
for sec in h.allsec():
    for seg in sec:
        seg.e_extracellular = 0  # Initialize extracellular potential

# Initialize the simulation
h.finitialize(-65)

# Apply the extracellular potential dynamically during simulation
def apply_extracellular():
    t_idx = int(h.t / h.dt) % len(stim_vec)
    for sec in h.allsec():
        for seg in sec:
            seg.e_extracellular = stim_vec.x[t_idx]

# Run simulation with dynamic update
h.tstop = 50  # Stop time in ms
while h.t < h.tstop:
    apply_extracellular()  # Apply extracellular potential dynamically
    h.fadvance()
