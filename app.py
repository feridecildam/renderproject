import streamlit as st
import gymnasium as gym
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --- SİTE AYARLARI ---
st.set_page_config(page_title="Frozen Lake AI Lab", layout="wide")

# --- 1. SİNİR AĞI MODELİ ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def state_to_tensor(state, state_size):
    tensor = torch.zeros(state_size)
    tensor[state] = 1.0
    return tensor

# --- SIDEBAR: PARAMETRELER VE MOD SEÇİMİ ---
st.sidebar.title("🛠️ Kontrol Merkezi")
app_mode = st.sidebar.selectbox("Uygulama Modu", ["🧠 DQN Eğitimi & Analiz", "🎮 Manuel/Single Step Kontrol"])

grid_size = st.sidebar.selectbox("Harita Boyutu", ["4x4", "8x8"])
is_slippery = st.sidebar.checkbox("Kaygan Zemin (Slippery)", value=False)

# --- MOD 1: DQN EĞİTİMİ VE ANALİZ ---
if app_mode == "🧠 DQN Eğitimi & Analiz":
    st.header("Deep Q-Learning Eğitim Laboratuvarı")
    
    col_params1, col_params2 = st.columns(2)
    with col_params1:
        episodes = st.number_input("Bölüm (Episode) Sayısı", 100, 2000, 1000, 100)
    with col_params2:
        gamma = st.slider("Gamma (Gelecek Odaklılık)", 0.1, 1.0, 0.95)

    if st.button("🚀 Eğitimi Başlat"):
        env = gym.make("FrozenLake-v1", map_name=grid_size, is_slippery=is_slippery, render_mode="rgb_array")
        state_size = env.observation_space.n
        action_size = env.action_space.n

        model = DQN(state_size, action_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        memory = deque(maxlen=2000)
        batch_size = 32
        epsilon, epsilon_decay, min_epsilon = 1.0, 0.995, 0.01
        
        reward_history = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for episode in range(episodes):
            state, _ = env.reset()
            terminated, truncated = False, False
            total_reward = 0
            
            while not (terminated or truncated):
                state_tensor = state_to_tensor(state, state_size)
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = torch.argmax(model(state_tensor)).item()
                
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                # Ödül Sistemi İyileştirmesi (Köşeye takılmayı önler)
                if terminated and reward == 0:
                    reward = -1 # Deliğe düştü
                elif new_state == state:
                    reward = -0.1 # Duvara çarptı (Köşeye takılma cezası)

                memory.append((state, action, reward, new_state, terminated or truncated))
                state = new_state
                total_reward += reward

                if len(memory) >= batch_size:
                    batch = random.sample(memory, batch_size)
                    b_states = torch.stack([state_to_tensor(s[0], state_size) for s in batch])
                    b_actions = torch.tensor([s[1] for s in batch]).unsqueeze(1)
                    b_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
                    b_next_states = torch.stack([state_to_tensor(s[3], state_size) for s in batch])
                    b_dones = torch.tensor([s[4] for s in batch], dtype=torch.float32)

                    current_q = model(b_states).gather(1, b_actions).squeeze()
                    max_next_q = model(b_next_states).max(1)[0]
                    targets = b_rewards + (gamma * max_next_q * (1 - b_dones))
                    
                    loss = loss_fn(current_q, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            reward_history.append(1 if total_reward > 0 else 0)
            if (episode + 1) % 10 == 0:
                progress_bar.progress((episode + 1) / episodes)

        st.success("✅ Eğitim Tamamlandı!")
        
        # Grafik
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(pd.Series(reward_history).rolling(50).mean(), color='green')
        ax.set_title("Eğitim Başarı Oranı (Son 50 Bölüm Ortalaması)")
        st.pyplot(fig)
        
        # Eğitilen modeli session_state'e kaydet (Mod 2'de kullanmak için)
        st.session_state.trained_model = model

# --- MOD 2: MANUEL / SINGLE STEP KONTROL ---
elif app_mode == "🎮 Manuel/Single Step Kontrol":
    st.header("Single Step & Manuel İzleme")
    
    if 'env' not in st.session_state or st.sidebar.button("♻️ Ortamı Yenile"):
        st.session_state.env = gym.make("FrozenLake-v1", map_name=grid_size, is_slippery=is_slippery, render_mode="rgb_array")
        st.session_state.state, _ = st.session_state.env.reset()
        st.session_state.game_over = False
        st.session_state.last_reward = 0

    col_game, col_info = st.columns([1, 1])

    with col_game:
        # Görseli küçültmek için container_width False yapıldı ve width ayarlandı
        frame = st.session_state.env.render()
        st.image(frame, caption="Frozen Lake Dünyası", width=350)
        
        st.write("### Hareket Kontrolü")
        c1, c2, c3 = st.columns(3)
        with c2: up = st.button("⬆️")
        with c1: left = st.button("⬅️")
        with c2: down = st.button("⬇️")
        with c3: right = st.button("➡️")
        
        action = None
        if left: action = 0
        elif down: action = 1
        elif right: action = 2
        elif up: action = 3

        # Eğer eğitilmiş model varsa yapay zeka adımını tetikle
        if st.button("🤖 Yapay Zeka Adımı At (DQN)"):
            if 'trained_model' in st.session_state:
                with torch.no_grad():
                    s_tensor = state_to_tensor(st.session_state.state, st.session_state.env.observation_space.n)
                    action = torch.argmax(st.session_state.trained_model(s_tensor)).item()
            else:
                st.warning("Önce eğitim modunda modeli eğitmelisiniz!")

        if action is not None and not st.session_state.game_over:
            n_state, rew, term, trunc, _ = st.session_state.env.step(action)
            st.session_state.state = n_state
            st.session_state.last_reward = rew
            st.session_state.game_over = term or trunc
            st.rerun()

    with col_info:
        st.subheader("📊 Anlık Veriler")
        st.metric("Mevcut Konum (State)", st.session_state.state)
        st.metric("Son Alınan Ödül", st.session_state.last_reward)
        
        if st.session_state.game_over:
            if st.session_state.last_reward > 0:
                st.success("🏆 HEDEFE ULAŞILDI!")
                st.balloons()
            else:
                st.error("💥 BOĞULDU! (Deliğe düştünüz)")
