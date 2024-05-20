import datetime
import pathlib
import random
import os
import asyncio
import aiofiles
import numpy as np
import orjson
from abc import ABC, abstractmethod
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0
        self.max_num_gpus = None

        ### Game
        self.observation_shape = (1, 1, 4)  # Game Observation (bp_systolic, bp_diastolic, heart_rate, respiratory_rate)
        self.action_space = list(
            range(2))  # Fixed list of all possible actions. You should only edit the length (good, bad)
        self.players = list(range(2))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0  # Turn MuZero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It
        # doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 9
        self.num_simulations = 25
        self.discount = 1
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"
        self.support_size = 10

        # Residual Network
        self.downsample = False
        self.blocks = 1
        self.channels = 16
        self.reduced_channels_reward = 16
        self.reduced_channels_value = 16
        self.reduced_channels_policy = 16
        self.resnet_fc_reward_layers = [8]
        self.resnet_fc_value_layers = [8]
        self.resnet_fc_policy_layers = [8]

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [16]
        self.fc_reward_layers = [16]
        self.fc_value_layers = []
        self.fc_policy_layers = []

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(
            __file__).stem / datetime.datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 2500
        self.batch_size = 64
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.003
        self.lr_decay_rate = 1
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 3000
        self.num_unroll_steps = 20
        self.td_steps = 20
        self.PER = True
        self.PER_alpha = 0.5

        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
                Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as
                training progresses. The smaller it is, the more likely the best action (ie with the highest visit count)
                 is chosen.

                Returns:
                    Positive float.
                """
        return 1


class Game(AbstractGame):
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

        # Display issues if any
        if self.env.issues:
            print("\nIssues with the current observation:")
            for issue in self.env.issues:
                print(issue)

        return observation, reward, done

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
        print("\n", end="")
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        print("\nReview the patient data above.")
        choice = input("Enter your decision (good or bad): ").strip().lower()
        while choice not in ["good", "bad"]:
            print("Invalid input. Please choose 'good' or 'bad'.")
            choice = input("Enter your decision (good or bad): ").strip().lower()
        return 1 if choice == "good" else 0

    def expert_agent(self):
        """
                Hard coded agent that MuZero faces to assess his progress in multiplayer games.
                It doesn't influence training

                Returns:
                    Action as an integer to take in the current game state
                """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        action_descriptions = {
            1: "Patient condition is good.",
            0: "Patient condition is bad.",
        }
        return action_descriptions.get(action_number, "Unknown action")


class PatientGame:
    def __init__(self, directory_path=r"D:\Old Files\Odyssey Therapiea\Muzero files\patient", seed=None):
        super().__init__()
        self.directory_path = directory_path
        self.patient_records = []
        self.current_record = None
        self.issues = None
        self.observations = {
            "good": 0,
            "bad": 1
        }
        self.score = 0
        asyncio.run(self.load_patient_data())

    async def load_patient_data(self):
        """ Load patient data asynchronously and shuffle the list of records. """
        json_files = [os.path.join(self.directory_path, file) for file in os.listdir(self.directory_path) if
                      file.endswith('.json')]
        for file_path in json_files:
            data = await self.read_json_file(file_path)
            processed_data = self.get_observations(data)
            self.patient_records.append(processed_data)
        random.shuffle(self.patient_records)
        self.current_record = self.patient_records.pop() if self.patient_records else np.zeros((1, 1, 4))

    async def read_json_file(self, file_path):
        """ Helper function to read a JSON file asynchronously. """
        async with aiofiles.open(file_path, 'rb') as file:
            data = await file.read()
            return orjson.loads(data)

    def get_observations(self, data):
        """Extract observations based on LOINC codes."""
        if not isinstance(data, dict):
            raise TypeError("Expected data to be a dictionary")
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

        observation_values = np.zeros(4)  # Create an array of zeros for 4 observations

        # Assign each observation based on LOINC code to the respective index
        for observation in observations:
            code_info = next((coding for coding in observation.get('code', {}).get('coding', []) if
                              coding.get('code') in loinc_codes.values()), None)
            if code_info:
                index = list(loinc_codes.values()).index(code_info['code'])
                observation_values[index] = observation.get('valueQuantity', {}).get('value', 0)

        # Reshape to (1,1,4) before returning
        reshaped_observations = np.reshape(observation_values, (1, 1, 4))
        return reshaped_observations

    def step(self, action):
        """Evaluate the player's action, update score, and move to the next record."""
        """Awarded 200 points if obv is correct and -200 if obv is wrong"""
        correct, issues = self.evaluate_current_record(action)
        reward = 200 if correct else -200
        game_over = len(self.patient_records) == 0
        if not game_over:
            self.current_record = self.patient_records.pop()
        else:
            self.current_record = np.zeros((1, 1, 4))
        return self.current_record, reward, game_over

    def evaluate_current_record(self, action):
        """ Evaluate the current record based on the action taken. """

        normal_ranges = {
            "heart_rate": (60, 100),
            "hemoglobin": (13, 17),
            "respiratory_rate": (12, 16),
            "diastolic_blood_pressure": (60, 80),
            "systolic_blood_pressure": (90, 120)
        }
        self.issues = []
        observation_values = self.current_record[0, 0]

        # Check each observation against normal ranges
        systolic_blood_pressure, diastolic_blood_pressure, heart_rate, respiratory_rate = observation_values
        if not (90 <= systolic_blood_pressure <= 120):
            self.issues.append(f"Systolic blood pressure out of range: {systolic_blood_pressure}")
        if not (60 <= diastolic_blood_pressure <= 80):
            self.issues.append(f"Diastolic blood pressure out of range: {diastolic_blood_pressure}")
        if not (60 <= heart_rate <= 100):
            self.issues.append(f"Heart rate out of range: {heart_rate}")
        if not (12 <= respiratory_rate <= 16):
            self.issues.append(f"Respiratory rate out of range: {respiratory_rate}")

        # Determine if the observation is good or bad based on issues
        observation_good = len(self.issues) == 0

        # Correct if action matches the observation status
        correct = (action == 1 and observation_good) or (action == 0 and not observation_good)
        return correct, self.issues

    def reset(self):
        """ Reset the game by reloading and shuffling the patient data. """
        asyncio.run(self.load_patient_data())
        return self.current_record

    def render(self):
        """ Display the current record's observations. """
        labels = ["Systolic Blood Pressure", "Diastolic Blood Pressure", "Heart Rate", "Respiratory Rate"]
        if self.current_record is not None:
            print("\nCurrent record observations:")
            for i, label in enumerate(labels):
                print(f"{label}: {self.current_record[0, 0, i]}")
        else:
            print("\nNo current record to display.")

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
