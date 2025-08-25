import random
import numpy as np
from memory_replay import MemoryReplay
import torch
from dqn import DQN
from collections import namedtuple


class AgentDQN:
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        initial_epsilon: float = 0.9,
        epsilon_decay: float = 0.1,
        final_epsilon: float = 0.1,
        lr=1e-3,
        gamma=0.99,
    ):

        self.n_observations = n_observations
        self.n_actions = n_actions

        # nn

        self.dqn = DQN(n_observations, n_actions)
        self.target_dqn = DQN(n_observations, n_actions)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        # hyperparameters

        self.gamma = gamma
        self.lr = lr

        # learning

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)

        self.loss_fn = torch.nn.SmoothL1Loss(reduction="none")

        # replay memory

        self.BATCH_SIZE = 32
        self.replay_memory = MemoryReplay(self.BATCH_SIZE)

        # epsilon greedy

        self.initial_epsilon = initial_epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def choisir_action(self, state: list[int]) -> int:
        if random.random() < self.epsilon:
            # action aléatoire
            res: int = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                t_actions = self.dqn.forward(torch.tensor(state, dtype=torch.float32))
                # res = t_actions.max(dim=1).indices.view(1, 1).item()
                res: int = torch.argmax(t_actions).item()  # type: ignore

        return res

    def memorize(
        self,
        state: list[int],
        action: int,
        reward: int,
        next_state: list[int],
        done: bool,
    ):

        if state[self.n_observations - 1] == 1:
            raise Exception("final state")

        tuple_experience = (
            state,
            action,
            reward,
            next_state,
            done,
        )

        self.replay_memory.append(tuple_experience)

    def batch_backward(self, batch: list, weights=None):
        transposee = [([]) for _ in range(5)]
        for experience in batch:
            for ix, item in enumerate(experience):
                transposee[ix].append(item)

        # tenseur de state (matrice N*n_observation)
        batch_state = torch.tensor(transposee[0], dtype=torch.float32)

        # tenseur d'action (matrice N*1)
        batch_action = torch.tensor(transposee[1]).unsqueeze(-1)

        # tenseur des prédictions (matrice N*n_action), gradient
        pred: torch.Tensor = self.dqn(batch_state)

        # matrice N*1 avec les QValues de (s, a), gradient
        state_action_values = pred.gather(1, batch_action)

        # tenseur de reward (matrice N*1)
        batch_reward = torch.tensor(transposee[2]).unsqueeze(-1)

        # matrice (M, 1) avec M <= N (car on retire les états finaux)
        batch_next_state_not_final = torch.tensor(
            [experience[3] for experience in batch if not experience[4]],
            dtype=torch.float32,
        )

        # vecteur taille N
        batch_is_not_final = torch.tensor(
            [not b for b in transposee[4]], dtype=torch.bool
        )

        # target : reward si état terminal, sinon reward + gamma * max(q_s'_a)
        next_state_values = torch.zeros(self.BATCH_SIZE)  # vecteur N

        if batch_next_state_not_final.shape[0] != 0:
            with torch.no_grad():
                res = self.target_dqn(batch_next_state_not_final).max(1).values
                next_state_values[batch_is_not_final] = res

        target = batch_reward + self.gamma * next_state_values.unsqueeze(-1)

        # retropropagation

        # loss = self.loss_fn(state_action_values, target)
        if weights is None:
            t_weights = torch.ones(len(batch)).unsqueeze(-1)
        else:
            t_weights = torch.tensor(weights)

        assert (
            t_weights.shape == state_action_values.shape
        ), "les poids doivent être de dimension (N, 1)"

        td_error = torch.abs(state_action_values - target).detach()
        # loss = torch.mean(t_weights * torch.pow(state_action_values - target, 2))
        loss_not_reduced = t_weights * self.loss_fn(state_action_values, target)

        assert (
            loss_not_reduced.shape == state_action_values.shape
        ), "la loss function ne doit pas reduire"

        loss = torch.mean(loss_not_reduced)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # on retourne l'erreur en valeur absolue pour chaque experience
        return td_error.numpy()

    def optimize_model(self):

        if len(self.replay_memory) < self.BATCH_SIZE:
            return

        batch = self.replay_memory.sample(self.BATCH_SIZE)
        td_error = self.batch_backward(batch)

        return np.mean(td_error)

    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def save_model(self, filepath):
        torch.save(self.dqn.model.state_dict(), filepath)

    def load_model(self, filepath):
        weights = torch.load(filepath, weights_only=True)
        self.dqn.model.load_state_dict(weights)
