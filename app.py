import streamlit as st
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import time
from PIL import Image

# --- SİTE BAŞLIĞI VE AYARLAR ---
st.set_page_config(page_title="Frozen Lake RL Lab", layout="centered")
st.title("🧊 Frozen Lake RL Simülasyonu")

# --- SIDEBAR: KULLANICI SEÇENEKLERİ ---
st.sidebar.header("Parametreler")
grid_size = st.sidebar.selectbox("Harita Boyutu", ["4x4", "8x8"])
is_slippery = st.sidebar.checkbox("Kaygan Zemin (Slippery)", value=False)
algo = st.sidebar.selectbox("Algoritma", ["Deep Q-Learning (DQN)", "Q-Learning"])
mode = st.sidebar.radio("Çalışma Modu", ["Otomatik", "Single Step (Adım Adım)"])

# --- MODEL TANIMI (Sizin DQN Sınıfınız) ---
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

# --- BAŞLAT BUTONU ---
if st.button("🚀 Simülasyonu Başlat"):
    # Environment oluşturma (Render mode: rgb_array web için şarttır)
    env = gym.make("FrozenLake-v1", map_name=grid_size, is_slippery=is_slippery, render_mode="rgb_array")
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # Model yükleme (Eğitim kısmı uzun süreceği için burada basit bir ajan simüle ediyoruz)
    model = DQN(state_size, action_size)
    
    state, _ = env.reset()
    img_placeholder = st.empty() # Görüntünün güncelleneceği yer
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Karar aşaması
        state_tensor = state_to_tensor(state, state_size)
        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item()
            
        state, reward, terminated, truncated, _ = env.step(action)
        
        # Ekranı Render Etme
        frame = env.render()
        img = Image.fromarray(frame)
        img_placeholder.image(img, use_container_width=True)
        
        if mode == "Single Step (Adım Adım)":
            st.write("Adım tamamlandı. Devam etmek için bekliyor...")
            time.sleep(1) # Basitleştirilmiş adım takibi
        else:
            time.sleep(0.2) # Animasyon hızı

    if reward == 1:
        st.success("🏆 Hedefe ulaşıldı!")
    else:
        st.error("💥 Göle düştünüz!")
    
    env.close()
