# -*- coding: utf-8 -*-
"""
FrozenLake RL Projesi - DQN + Tabular Q-Learning + PBRS
"""

import os
import random
from collections import deque
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, request
from flask_cors import CORS  # <-- EKLENDİ: Tarayıcı hatalarını (Failed to fetch) çözer

# =====================================================================
# 1. ORTAM VE ÖDÜL ŞEKİLLENDİRME (PBRS Wrapper)
# =====================================================================
class DinamikPotansiyelOdul:
    def __init__(self, izgara_boyutu, hedef_koordinati, w_hedef, w_delik, gamma=0.99):
        self.izgara_boyutu = izgara_boyutu
        self.hedef_koordinati = hedef_koordinati
        self.w_hedef = w_hedef
        self.w_delik = w_delik
        self.gamma = gamma
        self.kesfedilen_delikler = []
        self.max_mesafe = (izgara_boyutu[0] - 1) + (izgara_boyutu[1] - 1)

    def manhattan_mesafesi(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def potansiyel_hesapla(self, ajan_koordinati):
        mesafe_hedef = self.manhattan_mesafesi(ajan_koordinati, self.hedef_koordinati)
        hedef_potansiyeli = self.w_hedef * (self.max_mesafe - mesafe_hedef)
        delik_potansiyeli = 0
        if len(self.kesfedilen_delikler) > 0:
            mesafeler = [self.manhattan_mesafesi(ajan_koordinati, delik) for delik in self.kesfedilen_delikler]
            min_delik_mesafesi = min(mesafeler)
            delik_potansiyeli = self.w_delik * min_delik_mesafesi
        return hedef_potansiyeli + delik_potansiyeli

    def yeni_delik_ogren(self, delik_koordinati):
        if delik_koordinati not in self.kesfedilen_delikler:
            self.kesfedilen_delikler.append(delik_koordinati)

class PBRSFrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env, w_hedef, w_delik, gamma=0.99):
        super().__init__(env)
        self.w_hedef = w_hedef
        self.w_delik = w_delik
        self.satirlar = getattr(env.unwrapped, 'nrow', 4)
        self.sutunlar = getattr(env.unwrapped, 'ncol', 4)
        izgara_boyutu = (self.satirlar, self.sutunlar)
        hedef_koordinati = (self.satirlar - 1, self.sutunlar - 1)

        self.sekillendirici = DinamikPotansiyelOdul(
            izgara_boyutu=izgara_boyutu,
            hedef_koordinati=hedef_koordinati,
            w_hedef=self.w_hedef,
            w_delik=self.w_delik,
            gamma=gamma
        )
        self.eski_koordinat = None

    def duruma_koordinat_ver(self, state):
        return (int(state) // self.sutunlar, int(state) % self.sutunlar)

    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            state, info = reset_result
        else:
            state = reset_result
            info = {}
        self.eski_koordinat = self.duruma_koordinat_ver(state)
        return state, info

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
        else:
            next_state, reward, done, info = step_result
            terminated = done
            truncated = False

        yeni_koordinat = self.duruma_koordinat_ver(next_state)
        phi_eski = self.sekillendirici.potansiyel_hesapla(self.eski_koordinat)

        if terminated and float(reward) == 0.0:
            self.sekillendirici.yeni_delik_ogren(yeni_koordinat)

        phi_yeni = 0.0 if terminated else self.sekillendirici.potansiyel_hesapla(yeni_koordinat)
        ek_odul = (self.sekillendirici.gamma * phi_yeni) - phi_eski
        sekillendirilmis_odul = float(reward) + ek_odul
        self.eski_koordinat = yeni_koordinat

        return next_state, sekillendirilmis_odul, terminated, truncated, info

# =====================================================================
# 2. DEEP Q-LEARNING (DQN) BİLEŞENLERİ
# =====================================================================
class QNetwork(nn.Module):
    def __init__(self, durum_sayisi, eylem_sayisi):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(durum_sayisi, 64)
        self.fc2 = nn.Linear(64, 64)
        self.cikis = nn.Linear(64, eylem_sayisi)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.cikis(x)

class ReplayBuffer:
    def __init__(self, kapasite=10000):
        self.hafiza = deque(maxlen=kapasite)

    def ekle(self, durum, eylem, odul, sonraki_durum, bitti_mi):
        self.hafiza.append((durum, eylem, odul, sonraki_durum, bitti_mi))

    def orneklem_al(self, batch_size):
        return random.sample(self.hafiza, batch_size)

    def __len__(self):
        return len(self.hafiza)

class DQNAgent:
    def __init__(self, durum_sayisi, eylem_sayisi, lr=0.001, gamma=0.99):
        self.durum_sayisi = durum_sayisi
        self.eylem_sayisi = eylem_sayisi
        self.gamma = gamma
        self.batch_size = 64

        self.q_agi = QNetwork(durum_sayisi, eylem_sayisi)
        self.hedef_ag = QNetwork(durum_sayisi, eylem_sayisi)
        self.hedef_ag.load_state_dict(self.q_agi.state_dict())
        self.hedef_ag.eval()

        self.optimizer = optim.Adam(self.q_agi.parameters(), lr=lr)
        self.hafiza = ReplayBuffer()

    def state_to_tensor(self, state):
        one_hot = np.zeros(self.durum_sayisi)
        one_hot[int(state)] = 1.0
        return torch.FloatTensor(one_hot).unsqueeze(0)

    def eylem_sec(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.eylem_sayisi - 1)
        with torch.no_grad():
            q_degerleri = self.q_agi(self.state_to_tensor(state))
        return torch.argmax(q_degerleri).item()

    def ogren(self):
        if len(self.hafiza) < self.batch_size:
            return

        batch = self.hafiza.orneklem_al(self.batch_size)
        durumlar, eylemler, oduller, sonraki_durumlar, bitisler = zip(*batch)

        durumlar_tensor = torch.cat([self.state_to_tensor(s) for s in durumlar])
        sonraki_durumlar_tensor = torch.cat([self.state_to_tensor(s) for s in sonraki_durumlar])
        eylemler_tensor = torch.LongTensor(eylemler).unsqueeze(1)
        oduller_tensor = torch.tensor([float(o) for o in oduller], dtype=torch.float32).unsqueeze(1)
        bitisler_tensor = torch.tensor([float(b) for b in bitisler], dtype=torch.float32).unsqueeze(1)

        q_mevcut = self.q_agi(durumlar_tensor).gather(1, eylemler_tensor)

        with torch.no_grad():
            q_sonraki = self.hedef_ag(sonraki_durumlar_tensor).max(1)[0].unsqueeze(1)
            q_hedef = oduller_tensor + (self.gamma * q_sonraki * (1 - bitisler_tensor))

        loss = nn.MSELoss()(q_mevcut, q_hedef)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def hedef_agi_guncelle(self):
        self.hedef_ag.load_state_dict(self.q_agi.state_dict())

class agentBasedDQN:
    def __init__(self, environment, episode, w_hedef, w_delik):
        self.environment = PBRSFrozenLakeWrapper(env=environment, w_delik=w_delik, w_hedef=w_hedef)
        self.agent = DQNAgent(self.environment.observation_space.n, self.environment.action_space.n)
        self.episode = episode
        self.durum_sayisi = self.environment.observation_space.n

    def trainAgent(self):
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_azalma = 0.995
        ardisik_basari = 0
        path = []

        for bolum in range(self.episode):
            state, _ = self.environment.reset()
            done = False
            
            while not done:
                action = self.agent.eylem_sec(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated

                self.agent.hafiza.ekle(state, action, reward, next_state, done)
                self.agent.ogren()
                state = next_state

            if terminated:
                if int(state) == self.durum_sayisi - 1:
                    ardisik_basari += 1
                else:
                    ardisik_basari = 0

            if epsilon > epsilon_min:
                epsilon *= epsilon_azalma

            if bolum % 10 == 0:
                self.agent.hedef_agi_guncelle()

            if ardisik_basari >= 10:
                break

        # Test aşaması ve örnek yol çıkarma
        basari_sayisi = 0
        for i in range(5):
            state, _ = self.environment.reset()
            done = False
            current_path = [int(state)]
            while not done:
                action = self.agent.eylem_sec(state, 0.0)
                state, _, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                current_path.append(int(state))
                
                if terminated and int(state) == self.durum_sayisi - 1:
                    basari_sayisi += 1
                    
            if i == 0: # İlk testin yolunu al
                path = current_path

        return basari_sayisi, path, [] # DQN için q_table dönmüyoruz

# =====================================================================
# 3. TABULAR Q-LEARNING BİLEŞENLERİ
# =====================================================================
class QLearningAgent:
    def __init__(self, durum_sayisi, eylem_sayisi, lr=0.1, gamma=0.99):
        self.durum_sayisi = durum_sayisi
        self.eylem_sayisi = eylem_sayisi
        self.lr = lr
        self.gamma = gamma
        self.q_tablosu = np.zeros((durum_sayisi, eylem_sayisi))

    def eylem_sec(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.eylem_sayisi - 1)
        return int(np.argmax(self.q_tablosu[int(state), :]))

    def ogren(self, state, action, reward, next_state, done):
        state, next_state, action = int(state), int(next_state), int(action)
        max_gelecek_q = 0.0 if done else np.max(self.q_tablosu[next_state, :])
        mevcut_q = self.q_tablosu[state, action]
        yeni_q = mevcut_q + self.lr * (reward + self.gamma * max_gelecek_q - mevcut_q)
        self.q_tablosu[state, action] = yeni_q

class agentBasedQLearning:
    def __init__(self, environment, episode, w_hedef, w_delik):
        self.environment = PBRSFrozenLakeWrapper(env=environment, w_delik=w_delik, w_hedef=w_hedef)
        self.agent = QLearningAgent(self.environment.observation_space.n, self.environment.action_space.n)
        self.episode = episode
        self.durum_sayisi = self.environment.observation_space.n

    def trainAgent(self):
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_azalma = 0.995
        ardisik_basari = 0

        for bolum in range(self.episode):
            state, _ = self.environment.reset()
            done = False

            while not done:
                action = self.agent.eylem_sec(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                self.agent.ogren(state, action, reward, next_state, done)
                state = next_state

            if terminated:
                if int(state) == self.durum_sayisi - 1:
                    ardisik_basari += 1
                else:
                    ardisik_basari = 0

            if epsilon > epsilon_min:
                epsilon *= epsilon_azalma

            if ardisik_basari >= 10:
                break

        # Test aşaması ve örnek yol
        basari_sayisi = 0
        path = []
        for i in range(5):
            state, _ = self.environment.reset()
            done = False
            current_path = [int(state)]
            while not done:
                action = self.agent.eylem_sec(state, 0.0)
                state, _, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                current_path.append(int(state))
                
                if terminated and int(state) == self.durum_sayisi - 1:
                    basari_sayisi += 1
            if i == 0:
                path = current_path

        # Q Tablosunu listeye çevir (JSON serileştirmesi için)
        q_table_list = self.agent.q_tablosu.tolist()
        return basari_sayisi, path, q_table_list

# =====================================================================
# 4. YARDIMCI: Ortam oluşturucu
# =====================================================================
def ortam_olustur(harita="4x4", slippery=False):
    map_name = "4x4" if harita == "4x4" else "8x8"
    return gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery)

def parametreleri_al():
    try:
        w_hedef = float(request.args.get("w_hedef", 10.0))
    except ValueError:
        w_hedef = 10.0

    try:
        w_delik = float(request.args.get("w_delik", 10.0))
    except ValueError:
        w_delik = 10.0

    try:
        episode = int(request.args.get("episode", 500))
        episode = max(50, min(episode, 2000))
    except ValueError:
        episode = 500

    slippery_str = request.args.get("slippery", "false").lower()
    slippery = slippery_str in ("true", "1", "yes")

    harita = request.args.get("harita", "4x4")
    if harita not in ("4x4", "8x8"):
        harita = "4x4"

    return w_hedef, w_delik, episode, slippery, harita

# =====================================================================
# 5. FLASK WEB SUNUCUSU (API)
# =====================================================================
app = Flask(__name__)

# EKLENDİ: Tüm endpointlere her yerden erişim izni verir
CORS(app) 

@app.route("/")
def home():
    return jsonify({
        "mesaj": "FrozenLake RL API'sine Hos Geldiniz!",
        "endpointler": {
            "/train/dqn": "Deep Q-Network ile egitim",
            "/train/qlearning": "Tabular Q-Learning ile egitim",
            "/train/compare": "Her iki algoritmayi karsilastir",
        }
    })

@app.route("/train/dqn", methods=["GET"])
def train_dqn():
    w_hedef, w_delik, episode, slippery, harita = parametreleri_al()
    try:
        env = ortam_olustur(harita=harita, slippery=slippery)
        ajan = agentBasedDQN(environment=env, episode=episode, w_hedef=w_hedef, w_delik=w_delik)
        basari, sample_path, q_table = ajan.trainAgent()
        
        return jsonify({
            "durum": "basarili",
            "algoritma": "Deep Q-Network (DQN) + PBRS",
            "parametreler": {"harita": harita, "slippery": slippery},
            "success_count": basari,
            "total_tests": 5,
            "sample_path": sample_path,
            "q_table": q_table # DQN de boş döner
        })
    except Exception as e:
        return jsonify({"durum": "hata", "mesaj": str(e)}), 500

@app.route("/train/qlearning", methods=["GET"])
def train_qlearning():
    w_hedef, w_delik, episode, slippery, harita = parametreleri_al()
    try:
        env = ortam_olustur(harita=harita, slippery=slippery)
        ajan = agentBasedQLearning(environment=env, episode=episode, w_hedef=w_hedef, w_delik=w_delik)
        basari, sample_path, q_table = ajan.trainAgent()
        
        return jsonify({
            "durum": "basarili",
            "algoritma": "Tabular Q-Learning + PBRS",
            "parametreler": {"harita": harita, "slippery": slippery},
            "success_count": basari,
            "total_tests": 5,
            "sample_path": sample_path,
            "q_table": q_table # Arayüzde göstermek için Q Tablosu döner
        })
    except Exception as e:
        return jsonify({"durum": "hata", "mesaj": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
