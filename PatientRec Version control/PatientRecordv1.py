import datetime
import pathlib
import random
import os
import random
import asyncio
import aiofiles
import orjson
from abc import ABC, abstractmethod
import orjson
import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (1, 1,
                                  4)  # Game Observation (bp_systolic,bp_diastolic,heart_rate, respiratory_rate)
        self.action_space = list(range(2))  # Fixed list of all possible actions. You should only edit the length (good,bad)
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 9  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(
            __file__).stem / datetime.datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 2500  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = PatientGame()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        print("Review the patient data above.")
        choice = input("Enter your decision (good or bad): ").strip().lower()
        while choice not in self.legal_actions():
            print("Invalid input. Please choose 'good' or 'bad'.")
            choice = input("Enter your decision (good or bad): ").strip().lower()
        return choice

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """Convert an action number (or identifier) to a descriptive string."""
        action_descriptions = {
            'good': "Patient condition is good.",
            'bad': "Patient condition is bad.",
        }
        return action_descriptions.get(action_number, "Unknown action")


class PatientGame:
    def __init__(self, directory_path=r"D:\Old Files\Odyssey Therapiea\Muzero files\patient", seed=None):
        super().__init__()
        self.directory_path = directory_path
        self.patient_records = []
        self.current_record = None
        self.issues = None
        self.observations={
            "good":0,
            "bad":1
        }
        self.score = 0
        asyncio.run(self.load_patient_data())  # Load and shuffle patient data on initialization

    async def load_patient_data(self):
        """ Load patient data asynchronously and shuffle the list of records. """
        json_files = [os.path.join(self.directory_path, file) for file in os.listdir(self.directory_path) if
                      file.endswith('.json')]
        for file_path in json_files:
            data = await self.read_json_file(file_path)
            processed_data = self.get_observations(data)
            self.patient_records.append(processed_data)
        random.shuffle(self.patient_records)
        self.current_record = self.patient_records.pop() if self.patient_records else None

    async def read_json_file(self, file_path):
        """ Helper function to read a JSON file asynchronously. """
        async with aiofiles.open(file_path, 'rb') as file:
            data = await file.read()
            return orjson.loads(data)

    def get_observations(self, data):
        """Extract observations based on LOINC codes."""
        # Safely access the 'entry' key; default to an empty list if not found
        entries = data.get('entry', [])

        observations = [
            entry['resource'] for entry in entries
            if entry['resource']['resourceType'] == 'Observation'
        ]

        loinc_codes = {
            "heart_rate": "8867-4",
            "hemoglobin": "718-7",
            "respiratory_rate": "9279-1",
            "diastolic_blood_pressure": "8462-4",
            "systolic_blood_pressure": "8480-6"
        }

        filtered_observations = {}
        for observation in observations:
            if 'component' in observation:
                for component in observation['component']:
                    code = component['code']['coding'][0]['code']
                    if code in loinc_codes.values():
                        parameter = next(key for key, value in loinc_codes.items() if value == code)
                        filtered_observations[parameter] = {
                            "value": component['valueQuantity']['value'],
                            "unit": component['valueQuantity']['unit'],
                            "date": observation.get('effectiveDateTime')
                        }
            else:
                code_info = next((coding for coding in observation.get('code', {}).get('coding', []) if
                                  coding.get('code') in loinc_codes.values()), None)
                if code_info:
                    parameter = next(key for key, value in loinc_codes.items() if value == code_info['code'])
                    filtered_observations[parameter] = {
                        "value": observation.get('valueQuantity', {}).get('value'),
                        "unit": observation.get('valueQuantity', {}).get('unit'),
                        "date": observation.get('effectiveDateTime')
                    }

        return filtered_observations

    def step(self, action):
        """Evaluate the player's action, update score, and move to the next record."""
        correct, issues = self.evaluate_current_record(action, self.current_record)
        reward = 10 if correct else -10
        game_over = len(self.patient_records) == 0
        if not game_over:
            self.current_record = self.patient_records.pop()
        else:
            self.current_record = None
        return self.current_record, reward, game_over

    def evaluate_current_record(self, action, observations):
        """ Evaluate the current record based on the action taken. """
        normal_ranges = {
            "heart_rate": (60, 100),  # Example range, replace with actual
            "hemoglobin": (13, 17),  # Example range, replace with actual
            "respiratory_rate": (12, 16),  # Example range, replace with actual
            "diastolic_blood_pressure": (60, 80),  # Example range, replace with actual
            "systolic_blood_pressure": (90, 120)  # Example range, replace with actual
        }
        self.issues = []
        for parameter, (low, high) in normal_ranges.items():
            if parameter in observations:
                value = observations[parameter]['value']
                if not (low <= value <= high):
                    self.issues.append(f"{parameter} out of range: {value}")

        correct = (action == 'good' and not self.issues) or (action == 'bad' and self.issues)
        return correct, self.issues

    def reset(self):
        """ Reset the game by reloading and shuffling the patient data. """
        asyncio.run(self.load_patient_data())
        return self.get_observations(self.current_record)

    def render(self):
        """ Display the current record's observations. """
        print(f"Current record observations: {self.current_record}")

    def legal_actions(self):
        """ Returns the legal actions: 'good' or 'bad' """
        return [0, 1]

    def to_play(self):
        """ Single-player game, so always return player 0. """
        return 0

    def close(self):
        """ No resources to close in this example, but implement as needed. """
        pass

    def action_to_string(self, action_number):
        """ Convert action number to a descriptive string. """
        return "good" if action_number == 1 else "bad"