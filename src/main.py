from os import getcwd, path
from agent_dqn import AgentDQN
from environment import Environment
import torch
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import random

SHOULD_GENERATE_MAZE = True
SHOULD_TRAIN = True

N_EPISODE_TRAIN = 100

LAB_SIZE = 8


def main():
    n_observations = LAB_SIZE * LAB_SIZE
    n_actions = 4

    agent = AgentDQN(
        n_observations,
        n_actions,
        initial_epsilon=0.8,
        epsilon_decay=(0.8 - 0.1) / (0.5 * N_EPISODE_TRAIN),
        final_epsilon=0.1,
        lr=1e-2,
    )

    if SHOULD_GENERATE_MAZE:
        env = Environment(size=LAB_SIZE)
        env.save_labyrinthe(path.join(getcwd(), "labyrinthe.json"))
    else:
        env = Environment(labyrinthe_file_path=path.join(getcwd(), "labyrinthe.json"))

    if SHOULD_TRAIN:
        now = datetime.now()
        print(f"début de l'entraînement : {now}")

        train(agent, env)

        end = datetime.now()
        print(f"fin de l'entraînement : {end}")

        duration = end - now
        print(f"temps total d'entraînement : {duration}")

        agent.save_model(path.join(getcwd(), "model_checkpoint.ckpt"))
    else:
        agent.load_model(path.join(getcwd(), "model_checkpoint.ckpt"))

    list_all_q_values(agent)

    test(agent, env)


def int_state_to_list_state(int_state: int, n_observations: int) -> list[int]:
    state_list = [0 for _ in range(n_observations)]
    state_list[int_state] = 1
    return state_list


def list_all_q_values(agent: AgentDQN):
    all_states = torch.cat(
        [
            torch.tensor(
                int_state_to_list_state(state, agent.n_observations),
                dtype=torch.float32,
            ).unsqueeze(
                0
            )  # taille (1, n_actions)
            for state in range(agent.n_observations)
        ]
    )  # taille (n_observations, n_actions)
    with torch.no_grad():
        res = agent.dqn.model(all_states)

    print("================ final q values =====================")
    print(res)
    print("=====================================================")


def train(agent: AgentDQN, env: Environment):

    global_step = 0
    step_for_this_episode = 0

    episode_durations = []

    for episode in range(N_EPISODE_TRAIN):

        (observation,) = env.reset()
        state, liste_actions_possibles = observation

        done = False

        step_for_this_episode = 0
        while not done:

            state_as_list = int_state_to_list_state(state, agent.n_observations)

            action = agent.choisir_action(state_as_list)

            if action in liste_actions_possibles:
                observation, reward, done = env.step(action)
            else:
                observation, reward, done = observation, -10, False

            next_state, liste_actions_possibles = observation
            next_state_list = int_state_to_list_state(next_state, agent.n_observations)

            agent.memorize(
                state_as_list,
                action,
                reward,
                next_state_list,
                done,
            )

            agent.optimize_model()

            global_step += 1
            step_for_this_episode += 1

            if global_step % 25 == 0:
                agent.update_target_network()

            state = next_state

        episode_durations.append(step_for_this_episode)
        print(
            f"episode {episode}, {step_for_this_episode} steps, epsilon = {agent.epsilon}"
        )

        agent.decay_epsilon()
        episode += 1

    print(f"nombre total de steps : {global_step}")

    plt.title("Episode duration")  # type: ignore
    plt.xlabel("Épisode")  # type: ignore
    plt.ylabel("Duration")  # type: ignore
    plt.plot(torch.tensor(episode_durations, dtype=torch.int32).numpy())  # type: ignore
    # plt.savefig(  # type: ignore
    #     path.join(getcwd(), "mae.png"),
    # )
    plt.show()  # type: ignore


def test(agent: AgentDQN, env: Environment):

    (observation,) = env.reset()
    env.render()
    agent.set_epsilon(0)
    step = 0

    state, liste_actions_possibles = observation

    done = False
    while not done:

        state_as_list = int_state_to_list_state(state, agent.n_observations)

        action = agent.choisir_action(state_as_list)

        if action in liste_actions_possibles:
            observation, reward, done = env.step(action)
        else:
            observation, reward, done = observation, -1, False
        env.render()

        if env.quit_request:
            done = True

        next_state, liste_actions_possibles = observation

        state = next_state

        step += 1

    Image.fromarray(env.img).save(path.join(getcwd(), "maze.png"))


if __name__ == "__main__":
    main()
