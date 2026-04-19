from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import gymnasium as gym
import numpy as np
import random

app = FastAPI()

# CORS (frontend bağlanabilsin diye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HOLE_MAP = {
    "4x4": [5, 7, 11, 12],
    "8x8": [19, 29, 35, 41, 42, 46, 49, 52, 54, 59]
}

def calc_potential(state, size, w_hedef, w_delik):
    row, col = state // size, state % size
    goal_row, goal_col = size - 1, size - 1
    max_dist = (size - 1) * 2

    dist_goal = abs(row - goal_row) + abs(col - goal_col)
    pot = w_hedef * (max_dist - dist_goal)

    map_type = "4x4" if size == 4 else "8x8"
    holes = HOLE_MAP[map_type]

    if holes:
        min_hole_dist = float('inf')
        for h in holes:
            hr, hc = h // size, h % size
            d = abs(row - hr) + abs(col - hc)
            if d < min_hole_dist:
                min_hole_dist = d
        pot -= w_delik * (max_dist - min(min_hole_dist, max_dist))

    return pot


@app.get("/")
def home():
    return {"message": "API çalışıyor 🚀"}


@app.get("/train/{algorithm}")
def train_agent(
    algorithm: str,
    harita: str = Query("4x4"),
    slippery: str = Query("false"),
    episode: int = Query(500),
    w_hedef: float = Query(1.0),
    w_delik: float = Query(1.0)
):
    is_slippery = slippery.lower() == "true"
    size = 4 if harita == "4x4" else 8

    env = gym.make('FrozenLake-v1', map_name=harita, is_slippery=is_slippery)

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    if algorithm == "qlearning":
        for ep in range(episode):
            state, _ = env.reset()
            done = False
            truncated = False

            while not done and not truncated:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state])

                next_state, reward, done, truncated, _ = env.step(action)

                phi_old = calc_potential(state, size, w_hedef, w_delik)
                phi_new = calc_potential(next_state, size, w_hedef, w_delik)

                if done and next_state == env.observation_space.n - 1:
                    base_reward = 100
                elif done:
                    base_reward = -100
                else:
                    base_reward = 0

                shaped_reward = base_reward - 0.01 + (phi_new - phi_old)

                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                q_table[state, action] = old_value + alpha * (
                    shaped_reward + gamma * next_max - old_value
                )

                state = next_state

            epsilon = max(min_epsilon, epsilon * epsilon_decay)

    success_count = 0
    sample_path = []

    for _ in range(5):
        state, _ = env.reset()
        current_path = [int(state)]
        done = False
        truncated = False

        while not done and not truncated:
            action = np.argmax(q_table[state])
            state, _, done, truncated, _ = env.step(action)
            current_path.append(int(state))

            if done and state == env.observation_space.n - 1:
                success_count += 1
                if not sample_path or len(current_path) < len(sample_path):
                    sample_path = current_path

    if not sample_path:
        sample_path = current_path

    env.close()

    return {
        "durum": "basarili",
        "algoritma": f"{algorithm.upper()} + PBRS",
        "sonuc": f"{success_count}/5 basari",
        "sample_path": sample_path,
        "success_count": success_count
    }
