from __future__ import division

import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # GET RID OF TF DEPRECATION WARNINGS #
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from matplotlib import pyplot as plt
# %matplotlib inline
print(f"tf.__version__={tf.__version__}")
import numpy as np
import random

from psychrnn.backend.models.basic import Basic
from psychrnn.tasks.task import Task

# from psychrnn.tasks.perceptual_discrimination import PerceptualDiscrimination
class PerceptualDiscrimination(Task):
    """Two alternative forced choice (2AFC) binary discrimination task.

    On each trial the network receives two simultaneous noisy inputs into each of two input channels. The network must determine which channel has the higher mean input and respond by driving the corresponding output unit to 1.

    Takes two channels of noisy input (:attr:`N_in` = 2).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2) towards the higher mean channel.

    Loosely based on `Britten, Kenneth H., et al. "The analysis of visual motion: a comparison of neuronal and psychophysical performance." Journal of Neuroscience 12.12 (1992): 4745-4765 <https://www.jneurosci.org/content/12/12/4745>`_

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        coherence (float, optional): Amount by which the means of the two channels will differ. By default None.
        direction (int, optional): Either 0 or 1, indicates which input channel will have higher mean input. By default None.

    """

    def __init__(self, dt, tau, T, N_batch, coherence=None, direction=None):
        super(PerceptualDiscrimination, self).__init__(2, 2, dt, tau, T, N_batch)

        self.coherence = coherence

        self.direction = direction

        self.lo = 0.2  # Low value for one hot encoding

        self.hi = 1.0  # High value for one hot encoding

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys:
                * **coherence** (*float*) -- Amount by which the means of the two channels will differ. :attr:`self.coherence` if not None, otherwise ``np.random.exponential(scale=1/5)``.
                * **direction** (*int*) -- Either 0 or 1, indicates which input channel will have higher mean input. :attr:`self.direction` if not None, otherwise ``np.random.choice([0, 1])``.
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .1.
                * **onset_time** (*float*) -- Stimulus onset time. ``np.random.random() * self.T / 2.0``.
                * **stim_duration** (*float*) -- Stimulus duration. ``np.random.random() * self.T / 4.0 + self.T / 8.0``.

        """

        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        params = dict()
        if self.coherence == None:
            params['coherence'] = np.random.choice([0.1, 0.3, 0.5, 0.7])
        else:
            params['coherence'] = self.coherence
        params['direction'] = np.random.choice([0, 1])
        params['stim_noise'] = 0.1
        params['onset_time'] = np.random.random() * self.T / 2.0
        params['stim_duration'] = np.random.random() * self.T / 4.0 + self.T / 8.0

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. For ``params['onset_time'] < time < params['onset_time'] + params['stim_duration']`` , 1 is added to the noise in both channels, and :data:`params['coherence']` is also added in the channel corresponding to :data:`params[dir]`.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. From ``time > params['onset_time'] + params[stim_duration] + 20`` onwards, the correct output is encoded using one-hot encoding. Until then, y_t is 0 in both channels.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for ``time > params['onset_time'] + params['stim_duration']`` and False otherwise.

        """

        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(
            2 * .01 * np.sqrt(10) * np.sqrt(self.dt) * params['stim_noise'] * params['stim_noise']) * np.random.randn(
            self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        coh = params['coherence']
        onset = params['onset_time']
        stim_dur = params['stim_duration']
        dir = params['direction']

        # ----------------------------------
        # Compute values
        # ----------------------------------
        if onset < t < onset + stim_dur:
            x_t[dir] += 1 + coh
            x_t[(dir + 1) % 2] += 1

        if t > onset + stim_dur + 20:
            y_t[dir] = self.hi
            y_t[1 - dir] = self.lo

        if t < onset + stim_dur:
            mask_t = np.zeros(self.N_out)
        import pdb ; pdb.set_trace()
        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.

        """

        chosen = np.argmax(np.mean(test_output * output_mask, axis=1), axis=1)
        truth = np.argmax(np.mean(correct_output * output_mask, axis=1), axis=1)
        return np.mean(np.equal(truth, chosen))


seed=2020

tf.compat.v2.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

dt = 10 # The simulation timestep.
tau = 100 # The intrinsic time constant of neural state decay.
T = 2000 # The trial length.
N_batch = 50 # The number of trials per training update.
N_rec = 50 # The number of recurrent units in the network.
name = 'basicModel' #  Unique name used to determine variable scope for internal use.

pd = PerceptualDiscrimination(dt = dt, tau = tau, T = T, N_batch = N_batch) # Initialize the task object

network_params = pd.get_task_params()
print(network_params)

network_params['name'] = name # Unique name used to determine variable scope.
network_params['N_rec'] = N_rec # The number of recurrent units in the network.

network_params['rec_noise'] = 0.0 # Noise into each recurrent unit. Default: 0.0
network_params['W_in_train'] = True # Indicates whether W_in is trainable. Default: True
network_params['W_rec_train'] = True # Indicates whether W_rec is trainable. Default: True
network_params['W_out_train'] = True # Indicates whether W_out is trainable. Default: True
network_params['b_rec_train'] = True # Indicates whether b_rec is trainable. Default: True
network_params['b_out_train'] = True # Indicates whether b_out is trainable. Default: True
network_params['init_state_train'] = True # Indicates whether init_state is trainable. Default: True

network_params['transfer_function'] = tf.nn.relu # Transfer function to use for the network. Default: tf.nn.relu.
network_params['loss_function'] = "mean_squared_error"# String indicating what loss function to use. If not `mean_squared_error` or `binary_cross_entropy`, params["loss_function"] defines the custom loss function. Default: "mean_squared_error".

network_params['load_weights_path'] = None # When given a path, loads weights from file in that path. Default: None
# network_params['initializer'] = # Initializer to use for the network. Default: WeightInitializer (network_params) if network_params includes W_rec or load_weights_path as a key, GaussianSpectralRadius (network_params) otherwise.

network_params['which_rand_init'] = 'glorot_gauss' # Which random initialization to use for W_in and W_out. Will also be used for W_rec if which_rand_W_rec_init is not passed in. Options: 'const_unif', 'const_gauss', 'glorot_unif', 'glorot_gauss'. Default: 'glorot_gauss'.

network_params['which_rand_W_rec_init'] = network_params['which_rand_init'] # 'Which random initialization to use for W_rec. Options: 'const_unif', 'const_gauss', 'glorot_unif', 'glorot_gauss'. Default: which_rand_init.

network_params['init_minval'] = -.1 # Used by const_unif_init() as minval if 'const_unif' is passed in for which_rand_init or which_rand_W_rec_init. Default: -.1.

network_params['init_maxval'] = .1 # Used by const_unif_init() as maxval if 'const_unif' is passed in for which_rand_init or which_rand_W_rec_init. Default: .1.

network_params['L1_in'] = 0 # Parameter for weighting the L1 input weights regularization. Default: 0.

network_params['L1_rec'] = 0 # Parameter for weighting the L1 recurrent weights regularization. Default: 0.

network_params['L1_out'] = 0 # Parameter for weighting the L1 output weights regularization. Default: 0.

network_params['L2_in'] = 0 # Parameter for weighting the L2 input weights regularization. Default: 0.

network_params['L2_rec'] = 0 # Parameter for weighting the L2 recurrent weights regularization. Default: 0.

network_params['L2_out'] = 0 # Parameter for weighting the L2 output weights regularization. Default: 0.

network_params['L2_firing_rate'] = 0 # Parameter for weighting the L2 regularization of the relu thresholded states. Default: 0.

network_params['custom_regularization'] = None # Custom regularization function. Default: None.

basicModel = Basic(network_params)

train_params = {}
train_params['save_weights_path'] =  None # Where to save the model after training. Default: None
train_params['training_iters'] = 100000 # number of iterations to train for Default: 50000
train_params['learning_rate'] = .001 # Sets learning rate if use default optimizer Default: .001
train_params['loss_epoch'] = 10 # Compute and record loss every 'loss_epoch' epochs. Default: 10
train_params['verbosity'] = False # If true, prints information as training progresses. Default: True
train_params['save_training_weights_epoch'] = 100 # save training weights every 'save_training_weights_epoch' epochs. Default: 100
train_params['training_weights_path'] = None # where to save training weights as training progresses. Default: None
train_params['optimizer'] = tf.compat.v1.train.AdamOptimizer(learning_rate=train_params['learning_rate']) # What optimizer to use to compute gradients. Default: tf.train.AdamOptimizer(learning_rate=train_params['learning_rate'])
train_params['clip_grads'] = True # If true, clip gradients by norm 1. Default: True

train_params['fixed_weights'] = None # Dictionary of weights to fix (not allow to train). Default: None

train_params['performance_cutoff'] = None # If performance_measure is not None, training stops as soon as performance_measure surpases the performance_cutoff. Default: None.
train_params['performance_measure'] = None # Function to calculate the performance of the network using custom criteria. Default: None.]


losses, initialTime, trainTime = basicModel.train(pd, train_params)


plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Training Iteration")
plt.title("Loss During Training")

x,y,m, _ = pd.get_trial_batch()

plt.plot(range(0, len(x[0,:,:])*dt,dt), x[0,:,:])
plt.ylabel("Input Magnitude")
plt.xlabel("Time (ms)")
plt.title("Input Data")
plt.legend(["Input Channel 1", "Input Channel 2"])

output, state_var = basicModel.test(x)

plt.plot(range(0, len(output[0,:,:])*dt,dt),output[0,:,:])
plt.ylabel("Activity of Output Unit")
plt.xlabel("Time (ms)")
plt.title("Output on New Sample")
plt.legend(["Output Channel 1", "Output Channel 2"])

plt.plot(range(0, len(state_var[0,:,:])*dt,dt),state_var[0,:,:])
plt.ylabel("State Variable Value")
plt.xlabel("Time (ms)")
plt.title("Evolution of State Variables over Time")

weights = basicModel.get_weights()

print(weights.keys())

os.makedirs("/gpfs/milgram/scratch60/turk-browne/kp578/psychRNN/weights/", exist_ok=True)
basicModel.save("/gpfs/milgram/scratch60/turk-browne/kp578/psychRNN/weights/saved_weights")

basicModel.destruct()
