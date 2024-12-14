import matplotlib.pyplot as plt
from brian2 import *
from snn_constants import *


def plot_spikes(SM_E, SM_I=None, last_s=10):
  '''
  Arguments:
  SM_E: Voltages of excitatory neurons over course of training
  SM_I: Voltages of inhibitory neurons over course of training
  last_s: number of seconds to plot from end of training session

  Plots time given index of neuron for each spike
  '''
  plt.figure(figsize=(20, 6), dpi=80)

  e_e = SM_E.t[-1]/second
  idx = [i for i, v in enumerate(SM_E.t/second) if e_e - last_s <= v <= e_e]
  t_e = SM_E.t[idx]
  s_e = SM_E.i[idx]

  if SM_I:
    e_i = SM_I.t[-1]/second
    idx = [i for i, v in enumerate(SM_I.t/second) if e_i - last_s <= v <= e_i]
    t_i = SM_I.t[idx]
    s_i = SM_I.i[idx]

  plt.plot(t_e/ms, s_e, '.b', label='Excitatory')
  if SM_I:
    plt.plot(t_i/ms, s_i, '.r', label='Inhibitory')
  plt.xlabel('Time (ms)')
  plt.ylabel('Neuron index')
  plt.legend()
  plt.show()

def plot_vols(V_E, V_I, neurons):
  '''
  Arguments:
  V_E: Voltages of excitatory neurons over course of training
  V_I: Voltages of inhibitory neurons over course of training
  neurons: list of indices of excitatory and inhibitory neuron focused on

  Plots Time vs. Membrane voltage
  '''

  for i, n in enumerate(neurons):
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(V_E.t/ms, V_E.v[i]/mV, 'r', label='Excitatory')
    plt.plot(V_I.t/ms, V_I.v[i]/mV, 'b', label='Inhibitory')

    # Add horizontal bars to represent voltage thresholds for the two types of neurons
    plt.axhline(v_thresh_e/mV, color='green', label='v_thresh_e')
    plt.axhline(v_thresh_i/mV, color='black', label='v_thresh_i')

    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Voltage')
    plt.title('Neuron ' + str(n))
    plt.show()

def plot_stdp_vars(SM_STDP, neurons):
  '''
  Arguments:
  SM_STDP: Monitor containing values of STDP variables (w, Apre, Apost) over time
  neurons: list of indices of excitatory and inhibitory neuron focused on

  Return plot of SM_STDP
  '''
  plt.figure(figsize=(20, 10), dpi=80)

  plt.subplot(311)
  for i, n in enumerate(neurons):
    if i >= len(SM_STDP.w):
      break
    plt.plot(SM_STDP.t/ms, SM_STDP.w[i].T/gmax, label='E. ' + str(n))
  plt.legend()
  plt.ylabel('w/gmax')
  # plt.title('Input layer neuron 467 -> ???')

  plt.subplot(312)
  for i, n in enumerate(neurons):
    if i >= len(SM_STDP.w):
      break
    plt.plot(SM_STDP.t/ms, SM_STDP.Apre[i].T, label='E. ' + str(n))
  plt.legend()
  plt.ylabel('Apre')

  plt.subplot(313)
  for i, n in enumerate(neurons):
    if i >= len(SM_STDP.w):
      break
    plt.plot(SM_STDP.t/ms, SM_STDP.Apost[i].T, label='E. ' + str(n))
  plt.legend()
  plt.ylabel('Apost')
  plt.xlabel('Time (ms)')

  plt.show()

def visualize(network, neurons):
    '''
    Visualize network during training using plots of connected Brian monitors
    '''
    plot_stdp_vars(network.get('SM_STDP'), neurons)
    #r = np.random.randint(n_e, size=5)    #**************
    plot_vols(network.get('V_E'), network.get('V_I'), neurons)
    plot_spikes(network.get('SM_E'), network.get('SM_I'))

def plot_spikes_for(network, y_pred, y_true, I, n=5):
  '''
  Arguments:
    network
    y_pred: Predicted values returned by classifier
    y_true: Ground truth (correct) target values
    I: array of images
    n: number of images in test data to plot

    Plots time given index of neuron for each spike, for n images
  '''

  plt.figure(figsize=(n,n))

  # plot last n images in the training set with labels
  for i in range(1, n+1):
    plt.subplot(1, n, i)
    plt.imshow(I[-i], cmap=plt.get_cmap('gray'))
    plt.title('Pred: ' + str(y_pred[-i]))
    plt.axis('off')

  plt.show()

  # We use 0.5*n for last_s since each image is presented for 0.5 seconds (0.35 + 0.15)
  plot_spikes(network.get('SM_EVAL'), last_s=(0.5*n))