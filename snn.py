from brian2 import *


from brian2 import *
import numpy as np

from snn_constants import *

# This code is retrieved and slightly modified from the following Colab:
# https://colab.research.google.com/drive/1sLPjjMCmldwOxOQL5QPZMKqziVCBnga_?usp=sharing#scrollTo=c1IUIw8QjMnS


class SNN():
    def __init__(self, neurons, n_input, n_e, n_i):
        self.model = {}
        # model is a dictionary containing the names of and the compartments
        # in the neuron network.

        # Each input is a Poisson spike-train
        self.model['INPUT'] = PoissonGroup(n_input, rates=np.zeros(n_input)*Hz, name='INPUT')

        self.model['EXCITE'] = self.init_excit(n_e)
        self.model['INHIB'] = self.init_inhib(n_i)
        self.model['C1'] = self.conn_input_excit(self.model['INPUT'], self.model['EXCITE'])

        (C2, C3) = self.conn_inhib_excit(self.model['EXCITE'], self.model['INHIB'])
        self.model['C2'] =C2
        self.model['C3'] =C3

        # array of indices of neurons for monitoring
        self.neurons = neurons
        print('snn',self.neurons)

        # monitors = self.init_monitors(self.model)

        self.net = Network(list(self.model.values()))
        self.net.run(0*second)

    def get(self, val):

      return self.net[val]


    # Let students fill out the functions***************
    def init_excit(self, n_e):
        '''
        Initializes and returns the group of excitatory neurons using NeuronGroup()

        Read through section 2.1 for equations and
        1-intro-to-brian-neurons for usage on NeuronGroup()

        All needed parameters are given
        '''

        refrac_e = 5.*ms # Excitatory neuron refractory period

        neuron_e = '''
            dv/dt = (ge*(E_exc_e-v) + gi*(E_inh_e-v) + (v_rest_e-v)) / tau_e : volt
            dge/dt = -ge / tau_ge_e : 1
            dgi/dt = -gi / tau_gi_e : 1
            '''

        E = NeuronGroup(n_e, neuron_e, threshold='v>v_thresh_e', refractory=refrac_e, reset='v=v_reset_e', method='euler', name='EXCITE')
        E.v = v_init_e

        return E


    def init_inhib(self, n_i):

        '''
        Initializes and returns the group of inhibitory neurons.

        Analogous to init_excit()
        '''

        refrac_i = 2.*ms # Inhibitory neuron refractory period

        neuron_i = '''
            dv/dt = (ge*(E_exc_i-v) + (v_rest_i-v)) / tau_i : volt
            dge/dt = -ge / tau_ge_i : 1
            '''

        I = NeuronGroup(n_i, neuron_i, threshold='v>v_thresh_i', refractory=refrac_i, reset='v=v_reset_i', method='euler', name='INHIB')
        I.v = v_init_i

        return I


    def conn_input_excit(self, I, E):
        '''

        Arguments:
        I: input layer
        E: excitatory layer

        Initializes and returns the group of first-to-second layer synaptic connections of
        one-to-all connections between input and excitatory neurons.

        Use Synapses() for initialization, read through 2-intro-to-brian-synapses for usage on Synapses().

        All synapses from input neurons to excitatory neurons are
        learned using STDP.

        Follow usage of STDP model in end of 2-intro-to-brian-synapses
        and https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html.
        '''

        # Note addition of lr variable s.t
        # lr = 1 to enable stdp
        # lr = 0 to disable (weight unable to change)
        stdp='''w : 1
            lr : 1 (shared)
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)'''
        pre='''ge += w
            Apre += dApre
            w = clip(w + lr*Apost, 0, gmax)'''
        post='''Apost += dApost
            w = clip(w + lr*Apre, 0, gmax)'''

        C1 = Synapses(I, E, stdp, on_pre=pre, on_post=post, method='euler', name='C1')
        C1.connect()
        C1.w = 'rand()*gmax' # random weights initialization
        C1.lr = 1

        return C1

    def conn_inhib_excit(self, E, I):
        '''
        Initializes and returns the group of second layer synaptic connections consisting of:
            - One-to-one connections between excitatory and inhibitory neurons
            - One-to-all except for the one from which it receives a connection between inhibitory and excitatory neurons

        Description of Network Architecture in section 2.2.

        Returns the two synapse connections as a tuple (E->I, I->E)
        '''

        # excitatory neurons -> inhibitory neurons
        C2 = Synapses(E, I, 'w : 1', on_pre='ge += w', name='C2')
        C2.connect(j='i')
        C2.delay = 'rand()*10*ms'
        C2.w = 3 # strong weight to ensure an excitatory neuron will trigger a spike in its corresponding inhibitory neuron

        # inhibitory neurons -> excitatory neurons
        C3 = Synapses(I, E, 'w : 1', on_pre='gi += w', name='C3')
        C3.connect(condition='i!=j')
        C3.delay = 'rand()*5*ms'
        C3.w = .03

        return (C2, C3)

    def init_monitors(self, M):
        '''
        Initializes Brian2 monitors for graphing purposes

        Arguments:
        M: model of network containing neurons and synapses

        Adds a list of monitors including
            - Spike monitor for excitatory neurons
            - Spike monitor for inhibitory neurons
            - State monitor for excitatory neurons recording membrane voltage (v)
            - State monitor for inhibitory neurons recording membrane voltage (v)
            - State monitor for variables used in STDP (w, Apre, Apost)

        Descriptions and usage of Brian's SpikeMonitor() and StateMonitor() can be viewed
        at [1-intro-to-brian-neurons](https://brian2.readthedocs.io/en/stable/resources/tutorials/1-intro-to-brian-neurons.html)

        '''


        spikemonE = SpikeMonitor(M['EXCITE'], name='SM_E', record=True)
        # spikemonE = SpikeMonitor(M['EXCITE'][self.neurons_e], name='SM_E')
        # print('mon',self.neurons_e)
        spikemonI = SpikeMonitor(M['INHIB'], name='SM_I', record=True)
        # spikemonI = SpikeMonitor(M['INHIB'][self.neurons_e], name='SM_I')


        # Record membrane voltage for the two neuron groups
        volE = StateMonitor(M['EXCITE'], 'v', name='V_E', record=self.neurons)
        volI = StateMonitor(M['INHIB'], 'v', name='V_I', record=self.neurons)    # not needed changed to False

        # We record the input layer neuron connected to the center pixel of the 28x28 images: source 392
        # C1 connection from input layer to excitatory layer
        neurons = np.random.randint(100, size=4)        # placed outside neurons ro track
        stdp_mon = StateMonitor(M['C1'], ['w', 'Apre', 'Apost'], record=M['C1'][467, neurons], name='SM_STDP')

        self.net.add([spikemonE, spikemonI, volE, volI, stdp_mon])

    def train(self, X):
        '''
        Trains the model on a set of training images

        Arguments:
        X: List of training image data from dataset

        Follow the instructions for training in section 2.5 and 2.6
        '''
        self.net['C1'].lr = 1 # enable stdp

        print('... starting training ...')
        n_input = X.shape[1]
        for count, response_rates in enumerate(X[:-15]):
            
            # Present image to the network's input layer for 350 ms
            self.net['INPUT'].rates = response_rates * Hz
            self.net.run(duration_per_pattern)

            # Leave the network 150 ms without any input
            self.net['INPUT'].rates = np.zeros(n_input)*Hz
            self.net.run(duration_refractory)
            if np.mod(count,10) == 0:
              print(count)

        # construct monitors to store relevant network data
        # we start monitoring only until we have 15 images left in X to save memory
        self.init_monitors(self.net)

        for count, response_rates in enumerate(X[-15:]):
            # Present image to the network's input layer for 350 ms
            self.net['INPUT'].rates = response_rates * Hz
            self.net.run(duration_per_pattern)

            # Leave the network 150 ms without any input
            self.net['INPUT'].rates = np.zeros(n_input) * Hz
            self.net.run(duration_refractory)
            # print(counter)

    def evaluate(self, X, test=False):
        '''
        Evaluates the model after training is done

        Arguments:
        X: List of image data from dataset

        Follow the instructions for evaluation in section 2.5 and 2.6

        Returns:
        Spikes per neuron for each image
        '''
        self.net['C1'].lr = 0  # disable stdp

        if test:
            SM_EVAL = SpikeMonitor(self.net['EXCITE'], name='SM_EVAL')
            self.net.add(SM_EVAL)
        else:
            # Remove monitors used for training
            self.net.remove(self.net['SM_E'])
            self.net.remove(self.net['SM_I'])
            self.net.remove(self.net['V_E'])
            self.net.remove(self.net['V_I'])
            self.net.remove(self.net['SM_STDP'])

        # Spikes per neuron for each image
        spikes = []
        print(".... starting presentation/labeling phase ...")
        n_input = X.shape[1]

        for count, response_rates in enumerate(X):
            # Spike monitor to count number of excitatory neuron spikes for current image
            sm = SpikeMonitor(self.net['EXCITE'], name='MON')
            self.net.add(sm)

            # Present image to the network's input layer for 350 ms
            self.net['INPUT'].rates = response_rates * Hz
            self.net.run(duration_per_pattern)

            # Add spikes per neuron for current image
            spikes.append(np.array(sm.count, dtype=int8))

            # Leave the network 150 ms without any input
            self.net['INPUT'].rates = np.zeros(n_input)*Hz
            self.net.run(duration_refractory)

            # Remove monitor after presenting image (otherwise counts would accumulate)
            self.net.remove(self.net['MON'])
            if np.mod(count,10) == 0:
              print(count)

        return spikes