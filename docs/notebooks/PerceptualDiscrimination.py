"""
can be run in ood conda env named psychRNN

The next step is to modify the task to mimic my toy dataset to investigate NMPH.
"""
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


# from psychrnn.tasks.task import Task

# from psychrnn.tasks.perceptual_discrimination import PerceptualDiscrimination
from abc import ABCMeta, abstractmethod
ABC = ABCMeta('ABC', (object,), {})
class PerceptualDiscrimination(ABC):
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
        # super(PerceptualDiscrimination, self).__init__(2, 2, dt, tau, T, N_batch)

        # def __init__(self, N_in, N_out, dt, tau, T, N_batch):
        # ----------------------------------
        # Initialize required parameters
        # ----------------------------------
        N_in = 2
        N_out = 2
        self.N_batch = N_batch
        self.N_in = N_in
        self.N_out = N_out
        self.dt = dt
        self.tau = tau
        self.T = T

        # ----------------------------------
        # Calculate implied parameters
        # ----------------------------------
        self.alpha = (1.0 * self.dt) / self.tau
        self.N_steps = int(np.ceil(self.T / self.dt))

        # Initialize the generator used by get_trial_batch
        self._batch_generator = self.batch_generator()


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
        if onset < t < onset + stim_dur:  # If stimulus is on, add coherence to one channel
            x_t[dir] += 1 + coh
            x_t[(dir + 1) % 2] += 1

        if t > onset + stim_dur + 20:  # If stimulus is off, set correct output to one hot encoding
            y_t[dir] = self.hi
            y_t[1 - dir] = self.lo

        if t < onset + stim_dur:
            mask_t = np.zeros(self.N_out)
        # print(f"t={t}, x_t={x_t}, y_t={y_t}, mask_t={mask_t}")
        return x_t, y_t, mask_t

    def generate_trial(self, params):
        """ Loop to generate a single trial.

        Args:
            params(dict): Dictionary of trial parameters generated by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_trial** (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_in` *))*) -- Trial input given :data:`params`.
            * **y_trial** (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_out` *))*) -- Correct trial output given :data:`params`.
            * **mask_trial** (*ndarray(dtype=bool, shape=(*:attr:`N_steps`, :attr:`N_out` *))*) -- True during steps where the network should train to match :data:`y`, False where the network should ignore :data:`y` during training.
        """

        # ----------------------------------
        # Loop to generate a single trial
        # ----------------------------------
        x_data = np.zeros([self.N_steps, self.N_in])
        y_data = np.zeros([self.N_steps, self.N_out])
        mask = np.zeros([self.N_steps, self.N_out])

        for t in range(self.N_steps):
            x_data[t, :], y_data[t, :], mask[t, :] = self.trial_function(t * self.dt, params)
        # import pdb; pdb.set_trace()
        return x_data, y_data, mask

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.

        """

        chosen = np.argmax(np.mean(test_output * output_mask, axis=1), axis=1)
        truth = np.argmax(np.mean(correct_output * output_mask, axis=1), axis=1)
        # import pdb; pdb.set_trace()
        return np.mean(np.equal(truth, chosen))

    def batch_generator(self):
        """ Returns a generator for this task.

        Returns:
            Generator[tuple, None, None]:

        Yields:
            tuple:

            * **stimulus** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimuli for :attr:`N_batch` trials.
            * **target_output** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`stimulus`.
            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
            * **trial_params** (*ndarray(dtype=dict, shape =(*:attr:`N_batch` *,))*): Array of dictionaries containing the trial parameters produced by :func:`generate_trial_params` for each trial in :attr:`N_batch`.

        """

        batch = 1
        while batch > 0:

            x_data = []
            y_data = []
            mask = []
            params = []
            # ----------------------------------
            # Loop over trials in batch
            # ----------------------------------
            for trial in range(self.N_batch):  # self.N_batch = 50
                # ---------------------------------------
                # Generate each trial based on its params
                # ---------------------------------------
                p = self.generate_trial_params(batch, trial)
                x, y, m = self.generate_trial(p)
                x_data.append(x)
                y_data.append(y)
                mask.append(m)
                params.append(p)

            batch += 1
            # import pdb; pdb.set_trace()
            yield np.array(x_data), np.array(y_data), np.array(mask), np.array(params)  # yield is a keyword used in the context of defining a generator function
            # np.array(x_data).shape=(50, 200, 2)
            # np.array(y_data).shape=(50, 200, 2)
            # np.array(mask).shape=(50, 200, 2)
            # np.array(params).shape=(50,)  =
            # array([{'coherence': 0.1, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 308.74418390606695, 'stim_duration': 353.5978604610521},
            #        {'coherence': 0.1, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 197.6540435523988, 'stim_duration': 437.172076901579},
            #        {'coherence': 0.3, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 589.3771397379292, 'stim_duration': 508.2825230629569},
            #        {'coherence': 0.5, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 369.15766669837234, 'stim_duration': 565.0585224302322},
            #        {'coherence': 0.3, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 386.1472019348571, 'stim_duration': 347.307133023589},
            #        {'coherence': 0.5, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 226.96402941067984, 'stim_duration': 517.8371366721457},
            #        {'coherence': 0.1, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 683.5423818379871, 'stim_duration': 633.1751632164052},
            #        {'coherence': 0.1, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 817.3951461408548, 'stim_duration': 654.469767137837},
            #        {'coherence': 0.5, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 21.023850794471, 'stim_duration': 546.0887705614884},
            #        {'coherence': 0.1, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 992.9976240613183, 'stim_duration': 355.7911847698777},
            #        {'coherence': 0.5, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 647.0974917497293, 'stim_duration': 481.2903780449801},
            #        {'coherence': 0.3, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 161.80832139868295, 'stim_duration': 633.1723430658687},
            #        {'coherence': 0.1, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 54.48680547192808, 'stim_duration': 680.9729231554711},
            #        {'coherence': 0.1, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 468.56722756746836, 'stim_duration': 576.1391445402248},
            #        {'coherence': 0.5, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 146.86779687701767, 'stim_duration': 727.8850317380181},
            #        {'coherence': 0.7, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 130.58216575412175, 'stim_duration': 431.8263245790081},
            #        {'coherence': 0.3, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 755.5103538967599, 'stim_duration': 695.9531564166084},
            #        {'coherence': 0.3, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 822.9584523595978, 'stim_duration': 349.3089398824497},
            #        {'coherence': 0.1, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 318.62685118050115, 'stim_duration': 441.1258827103437},
            #        {'coherence': 0.7, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 720.0297268562983, 'stim_duration': 687.7259332994447},
            #        {'coherence': 0.1, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 50.35050642357608, 'stim_duration': 466.20959573877946},
            #        {'coherence': 0.1, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 990.3369047614766, 'stim_duration': 424.0966277380552},
            #        {'coherence': 0.7, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 908.1651166303327, 'stim_duration': 702.7392439155907},
            #        {'coherence': 0.1, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 365.99255288463263, 'stim_duration': 671.9702112640533},
            #        {'coherence': 0.3, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 626.515377531532, 'stim_duration': 331.9580532978897},
            #        {'coherence': 0.5, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 958.3602424014681, 'stim_duration': 395.1127329223317},
            #        {'coherence': 0.1, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 778.2117953042168, 'stim_duration': 706.714778568107},
            #        {'coherence': 0.5, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 530.0662043576103, 'stim_duration': 747.5744508906422},
            #        {'coherence': 0.7, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 303.69857873099085, 'stim_duration': 667.5773347759198},
            #        {'coherence': 0.3, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 128.20600676557137, 'stim_duration': 395.80635209172124},
            #        {'coherence': 0.7, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 192.65948625690655, 'stim_duration': 256.9066132552221},
            #        {'coherence': 0.1, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 114.61026275550445, 'stim_duration': 654.3300671197885},
            #        {'coherence': 0.5, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 873.1535104840648, 'stim_duration': 612.7292766953615},
            #        {'coherence': 0.7, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 994.0254997020534, 'stim_duration': 465.4883889098831},
            #        {'coherence': 0.3, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 269.9321518036368, 'stim_duration': 649.5714675951697},
            #        {'coherence': 0.5, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 740.1040840647987, 'stim_duration': 594.7134988503244},
            #        {'coherence': 0.7, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 387.12288652233696, 'stim_duration': 650.3179255653145},
            #        {'coherence': 0.5, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 729.7748511613954, 'stim_duration': 447.832682181059},
            #        {'coherence': 0.3, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 901.0577180829221, 'stim_duration': 620.871962695531},
            #        {'coherence': 0.5, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 914.6556861186349, 'stim_duration': 597.0827433081727},
            #        {'coherence': 0.3, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 503.8279994557727, 'stim_duration': 278.08852881259577},
            #        {'coherence': 0.7, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 486.1445726320894, 'stim_duration': 264.15134955198454},
            #        {'coherence': 0.5, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 126.25078876768924, 'stim_duration': 742.2921640738396},
            #        {'coherence': 0.7, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 471.02592272151634, 'stim_duration': 321.5884845514644},
            #        {'coherence': 0.3, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 340.5532802506712, 'stim_duration': 603.9346183478183},
            #        {'coherence': 0.3, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 944.3529376563898, 'stim_duration': 618.8949949459986},
            #        {'coherence': 0.5, 'direction': 1, 'stim_noise': 0.1, 'onset_time': 424.5013097112977, 'stim_duration': 617.0000700932467},
            #        {'coherence': 0.5, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 786.6755197118141, 'stim_duration': 637.0990829680006},
            #        {'coherence': 0.3, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 317.7992014669083, 'stim_duration': 564.9087448310853},
            #        {'coherence': 0.3, 'direction': 0, 'stim_noise': 0.1, 'onset_time': 426.46354976431036, 'stim_duration': 577.6885329128518}],
            #       dtype=object)

    def get_trial_batch(self):
        """Get a batch of trials.

        Wrapper for :code:`next(self._batch_generator)`.

        Returns:
            tuple:

            * **stimulus** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_in` *))*): Task stimuli for :attr:`N_batch` trials.
            * **target_output** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`stimulus`.
            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
            * **trial_params** (*ndarray(dtype=dict, shape =(*:attr:`N_batch` *,))*): Array of dictionaries containing the trial parameters produced by :func:`generate_trial_params` for each trial in :attr:`N_batch`.

        """
        return next(self._batch_generator)

    def get_task_params(self):
        """ Get dictionary of task parameters.

        Note:
            N_in, N_out, N_batch, dt, tau and N_steps must all be passed to the network model as parameters -- this function is the recommended way to begin building the network_params that will be passed into the RNN model.


        Returns:
            dict: Dictionary of :class:`Task` attributes including the following keys:

            :Dictionary Keys:
                * **N_batch** (*int*) -- The number of trials per training update.
                * **N_in** (*int*) -- The number of network inputs.
                * **N_out** (*int*) -- The number of network outputs.
                * **dt** (*float*) -- The simulation timestep.
                * **tau** (*float*) -- The unit time constant.
                * **T** (*float*) -- The trial length.
                * **alpha** (*float*) -- The number of unit time constants per simulation timestep.
                * **N_steps** (*int*): The number of simulation timesteps in a trial.

            Note:
                The dictionary will also include any other attributes defined in your task definition.

        """
        return self.__dict__


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


# from psychrnn.backend.models.basic import Basic
from psychrnn.backend.rnn import RNN
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
class Basic(RNN):
    """ The basic continuous time recurrent neural network model.

    Basic implementation of :class:`psychrnn.backend.rnn.RNN` with a simple RNN, enabling biological constraints.

    Args:
       params (dict): See :class:`psychrnn.backend.rnn.RNN` for details.

    """

    def recurrent_timestep(self, rnn_in, state):
        """ Recurrent time step.

        Given input and previous state, outputs the next state of the network.

        Arguments:
            rnn_in (*tf.Tensor(dtype=float, shape=(?*, :attr:`N_in` *))*): Input to the rnn at a certain time point.
            state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): State of network at previous time point.

        Returns:
            new_state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): New state of the network.

        """

        new_state = ((1 - self.alpha) * state) \
                    + self.alpha * (
                            tf.matmul(
                                self.transfer_function(state),
                                self.get_effective_W_rec(),
                                transpose_b=True, name="1")
                            + tf.matmul(
                        rnn_in,
                        self.get_effective_W_in(),
                        transpose_b=True, name="2")
                            + self.b_rec) \
                    + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise) \
                    * tf.random.normal(tf.shape(input=state), mean=0.0, stddev=1.0)

        return new_state

    def output_timestep(self, state):
        """Returns the output node activity for a given timestep.

        Arguments:
            state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): State of network at a given timepoint for each trial in the batch.

        Returns:
            output (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_out` *))*): Output of the network at a given timepoint for each trial in the batch.

        """

        output = tf.matmul(self.transfer_function(state),
                           self.get_effective_W_out(), transpose_b=True, name="3") \
                 + self.b_out

        return output

    def forward_pass(self):
        """ Run the RNN on a batch of task inputs.

        Iterates over timesteps, running the :func:`recurrent_timestep` and :func:`output_timestep`

        Implements :func:`psychrnn.backend.rnn.RNN.forward_pass`.

        Returns:
            tuple:
            * **predictions** (*tf.Tensor(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*) -- Network output on inputs found in self.x within the tf network.
            * **states** (*tf.Tensor(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_rec` *))*) -- State variable values over the course of the trials found in self.x within the tf network.

        """

        rnn_inputs = tf.unstack(self.x, axis=1)
        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        for rnn_input in rnn_inputs:
            state = self.recurrent_timestep(rnn_input, state)
            output = self.output_timestep(state)
            rnn_outputs.append(output)
            rnn_states.append(state)
        return tf.transpose(a=rnn_outputs, perm=[1, 0, 2]), tf.transpose(a=rnn_states, perm=[1, 0, 2])



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


plt.figure()
plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Training Iteration")
plt.title("Loss During Training")

x,y,m, _ = pd.get_trial_batch()

plt.figure()
plt.plot(range(0, len(x[0,:,:])*dt,dt), x[0,:,:])
plt.ylabel("Input Magnitude")
plt.xlabel("Time (ms)")
plt.title("Input Data")
plt.legend(["Input Channel 1", "Input Channel 2"])

output, state_var = basicModel.test(x)

plt.figure()
plt.plot(range(0, len(output[0,:,:])*dt,dt),output[0,:,:])
plt.ylabel("Activity of Output Unit")
plt.xlabel("Time (ms)")
plt.title("Output on New Sample")
plt.legend(["Output Channel 1", "Output Channel 2"])

plt.figure()
plt.plot(range(0, len(state_var[0,:,:])*dt,dt),state_var[0,:,:])
plt.ylabel("State Variable Value")
plt.xlabel("Time (ms)")
plt.title("Evolution of State Variables over Time")

weights = basicModel.get_weights()

print(weights.keys())

os.makedirs("/gpfs/milgram/scratch60/turk-browne/kp578/psychRNN/weights/", exist_ok=True)
basicModel.save("/gpfs/milgram/scratch60/turk-browne/kp578/psychRNN/weights/saved_weights")

# basicModel.destruct()
