import numpy as np
from brian2 import defaultclock, ms, second, mV

# Simulation parameters
defaultclock.dt = 10 * ms
dt = defaultclock.dt
duration = 100 * ms  # Duration of each pattern presentation

# ---------------------------- Constants for init_excit() ---------------------------------

# Excitatory neuron parameters for leaky integrate-and-fire model
v_rest_e = -60.*mV # resting membrane potential
v_reset_e = -65.*mV # resting membrane potential
v_thresh_e = -52.*mV # resting membrane threshold                 -52  lowered
v_init_e = v_rest_e - 20.*mV # initial membrane potential

# Equilibrium potentials
E_inh_e = -100.*mV
E_exc_e = 0*mV

# Time constants
tau_e = 100 * ms
tau_ge_e = 5*ms
tau_gi_e = 10*ms

duration_per_pattern = 10 * second
duration_refractory =  duration_per_pattern / 3
# ---------------------------- Constants for init_inhib() ------------------------------

# Inhibitory neuron parameters for leaky integrate-and-fire model
v_rest_i = -60.*mV # resting membrane potential
v_reset_i = -45.*mV # resting membrane potential
v_thresh_i = -40.*mV # resting membrane threshold
v_init_i = v_rest_i - 20.*mV # initial membrane potential

# Equilibrium potentials
E_inh_i = -85.*mV
E_exc_i = 0*mV

# Time constants
tau_i = 10 * ms
tau_ge_i = 5*ms
tau_gi_i = 2.0*ms

# --------------------------- Constants for stdp ------------------------------

taupre = taupost = 20*ms
gmax = .05
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax
