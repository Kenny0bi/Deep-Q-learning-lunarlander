import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import csv

def run(episodes, is_training=True, render=False):
    # Crea el entorno de LunarLander
    env = gym.make("LunarLander-v2", continuous=False, enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='human' if render else None)

    # Configura el espacio de estado (posición y velocidad) para LunarLander
    pos_space = np.linspace(-1, 1, 20)   # Rango simplificado para la posición en x (horizontal)
    vel_space = np.linspace(-2, 2, 20)   # Rango simplificado para la velocidad en x

    # Inicializa o carga la tabla Q
    if is_training or not os.path.exists('lunar_lander.pkl'):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        with open('lunar_lander.pkl', 'rb') as f:
            q = pickle.load(f)

    # Hiperparámetros
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 2 / episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    # Abre el archivo CSV para registrar las transiciones
    with open("lunar_lander_transitions.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["state_x", "state_y", "velocity_x", "velocity_y", "action", "reward", "next_state_x", "next_state_y", "next_velocity_x", "next_velocity_y", "done"])

        for i in range(episodes):
            state = env.reset()[0]
            state_p = min(np.digitize(state[0], pos_space) - 1, len(pos_space) - 1)  # Ajuste para índice de posición x
            state_v = min(np.digitize(state[1], vel_space) - 1, len(vel_space) - 1)  # Ajuste para índice de velocidad x

            terminated = False
            rewards = 0

            while not terminated and rewards > -1000:
                # Selección de acción
                if is_training and rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state_p, state_v, :])

                # Realiza la acción y observa el nuevo estado
                new_state, reward, terminated, _, _ = env.step(action)
                new_state_p = min(np.digitize(new_state[0], pos_space) - 1, len(pos_space) - 1)  # Índice para posición x del nuevo estado
                new_state_v = min(np.digitize(new_state[1], vel_space) - 1, len(vel_space) - 1)  # Índice para velocidad x del nuevo estado

                # Actualiza la tabla Q en modo de entrenamiento
                if is_training:
                    q[state_p, state_v, action] += learning_rate_a * (
                        reward + discount_factor_g * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                    )

                # Registra la transición en el archivo CSV
                writer.writerow([state[0], state[1], state[2], state[3], action, reward, new_state[0], new_state[1], new_state[2], new_state[3], terminated])

                state = new_state
                state_p = new_state_p
                state_v = new_state_v

                rewards += reward

            epsilon = max(epsilon - epsilon_decay_rate, 0)
            rewards_per_episode[i] = rewards

    env.close()

    # Guarda la tabla Q si es en modo de entrenamiento
    if is_training:
        with open('lunar_lander.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Grafica la recompensa promedio
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(mean_rewards)
    plt.savefig('lunar_lander.png')

if __name__ == '__main__':
    # Ejecuta el entrenamiento para crear el archivo
    run(5000, is_training=True, render=False)

    # Luego ejecuta en modo de evaluación
    #run(15, is_training=False, render=True)
