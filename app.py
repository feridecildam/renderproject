# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:39:19 2026

@author: Excalibur
"""

class DinamikPotansiyelOdul:
    def __init__(self, izgara_boyutu, hedef_koordinati, w_hedef, w_delik, gamma=0.99):
        self.izgara_boyutu = izgara_boyutu
        self.hedef_koordinati = hedef_koordinati
        self.w_hedef = w_hedef
        self.w_delik = w_delik
        self.gamma = gamma
        
        # Ajanın düşerek öğreneceği deliklerin listesi (Başlangıçta boş)
        self.kesfedilen_delikler = [] 
        
        # Haritadaki maksimum olası mesafe (potansiyeli ters çevirmek için)
        self.max_mesafe = (izgara_boyutu[0] - 1) + (izgara_boyutu[1] - 1)

    def manhattan_mesafesi(self, pos1, pos2):
        # Izgara üzerindeki dikey ve yatay adım mesafesini hesaplar
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def potansiyel_hesapla(self, ajan_koordinati):
        # 1. HEDEF BİLEŞENİ (Hedefe yaklaştıkça artar)
        mesafe_hedef = self.manhattan_mesafesi(ajan_koordinati, self.hedef_koordinati)
        hedef_potansiyeli = self.w_hedef * (self.max_mesafe - mesafe_hedef)

        # 2. DELİK BİLEŞENİ (Delikten uzaklaştıkça artar)
        delik_potansiyeli = 0
        if len(self.kesfedilen_delikler) > 0:
            # Ajanın bildiği tüm deliklere olan mesafesini ölç ve en yakınındakini bul
            mesafeler = [self.manhattan_mesafesi(ajan_koordinati, delik) for delik in self.kesfedilen_delikler]
            min_delik_mesafesi = min(mesafeler)
            
            # Deliğe mesafe arttıkça potansiyel yükselir (Ajanı delikten iter)
            delik_potansiyeli = self.w_delik * min_delik_mesafesi

        # Toplam durum potansiyeli
        return hedef_potansiyeli + delik_potansiyeli

    def yeni_delik_ogren(self, delik_koordinati):
        # Ajan deliğe düştüğünde bu fonksiyon çağrılır
        if delik_koordinati not in self.kesfedilen_delikler:
            self.kesfedilen_delikler.append(delik_koordinati)


import gymnasium as gym

class PBRSFrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env, w_hedef, w_delik, gamma=0.99):
        super().__init__(env)
        self.w_hedef = w_hedef
        self.w_delik = w_delik
        # 1. ORTAMDAN BOYUTLARI OTOMATİK ÇEKME
        # env.unwrapped ile sarıcıları aşıp ortamın asıl özelliklerine erişiyoruz
        self.satirlar = env.unwrapped.nrow
        self.sutunlar = env.unwrapped.ncol
        
        izgara_boyutu = (self.satirlar, self.sutunlar)
        
        # 2. HEDEFİ OTOMATİK BULMA
        # FrozenLake ortamında hedef daima en alt sağ köşededir (Örn: 4x4 için 3,3)
        hedef_koordinati = (self.satirlar - 1, self.sutunlar - 1)
        
        # 3. Şekillendiriciyi dinamik değerlerle başlat
        self.sekillendirici = DinamikPotansiyelOdul(
            izgara_boyutu=izgara_boyutu, 
            hedef_koordinati=hedef_koordinati,
            w_hedef=self.w_hedef,
            w_delik=self.w_delik,
            gamma=gamma
        )
        self.eski_koordinat = None

    def duruma_koordinat_ver(self, state):
        # Dönüşüm artık otomatik olarak ortamın genişliğine göre (self.sutunlar) yapılıyor
        return (state // self.sutunlar, state % self.sutunlar)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.eski_koordinat = self.duruma_koordinat_ver(state)
        return state, info

    def step(self, action):
        # 1. ORİJİNAL ÇEVREDEN ADIM AT
        next_state, reward, terminated, truncated, info = self.env.step(action)
        yeni_koordinat = self.duruma_koordinat_ver(next_state)
        
        # -------------------------------------------------------------
        # DİKKAT: DPBRS TEOREMİ GEREĞİ KRİTİK SIRALAMA VE TERMİNAL KONTROLÜ!
        # -------------------------------------------------------------
        
        # A) Eski durumun potansiyelini ESKİ BİLGİYLE (henüz yeni delik öğrenilmeden) hesapla
        phi_eski = self.sekillendirici.potansiyel_hesapla(self.eski_koordinat)
        
        # B) Eğer yeni bir delik keşfedildiyse, BİLGİYİ GÜNCELLE
        if terminated and reward == 0:
            self.sekillendirici.yeni_delik_ogren(yeni_koordinat)
            
        # C) Yeni durumun potansiyelini YENİ BİLGİYLE hesapla
        # TEOREM DÜZELTMESİ: Eğer oyun bittiyse (delik veya hedef), potansiyel SIFIRLANMALIDIR.
        if terminated:
            phi_yeni = 0.0
        else:
            phi_yeni = self.sekillendirici.potansiyel_hesapla(yeni_koordinat)
        
        # D) PBRS Formülünü uygula
        ek_odul = (self.sekillendirici.gamma * phi_yeni) - phi_eski
        # -------------------------------------------------------------

        sekillendirilmis_odul = reward + ek_odul
        self.eski_koordinat = yeni_koordinat
        
        return next_state, sekillendirilmis_odul, terminated, truncated, info
    
    

import DeepQLearning
import FrozenlakeWrapper

class agentBasedDQN:
    
    def __init__(self,environment,episode,w_hedef,w_delik):
        self.w_hedef = w_hedef
        self.w_delik = w_delik
        ### Environment driver koddan gelmeli.
        self.environment = FrozenlakeWrapper.PBRSFrozenLakeWrapper(env=environment,w_delik=self.w_delik,w_hedef=self.w_hedef) ##Ağırlıklar da güncellenmeli.
        self.agent = DeepQLearning.DQNAgent(self.environment.observation_space.n, self.environment.action_space.n)
        self.episode = episode
        self.durum_sayisi = self.environment.observation_space.n
        
    def trainAgent(self):
       
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_azalma = 0.995
        ardisik_basari = 0
        print("--- EĞİTİM BAŞLIYOR (Epsilon-Greedy) ---")
        for bolum in range(self.episode):
            state, _ = self.environment.reset()
            done = False
            toplam_odul = 0
            
            while not done:
                action = self.agent.eylem_sec(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                
                # (PBRS Wrapper kullanıyorsanız 'reward' otomatik şekillenmiş gelecektir)
                self.agent.hafiza.ekle(state, action, reward, next_state, done)
                self.agent.ogren()
                
                state = next_state
                toplam_odul += reward
                
                if terminated:
                    if state == self.durum_sayisi - 1: # Hedefe ulaştı
                        ardisik_basari += 1
                    else: # Deliğe düştü
                        ardisik_basari = 0
                
            # Epsilon'u yavaşça düşür (Keşiften sömürüye geçiş)
            if epsilon > epsilon_min:
                epsilon *= epsilon_azalma
                
            # Hedef ağı periyodik olarak güncelle
            if bolum % 10 == 0:
                self.agent.hedef_agi_guncelle()
            
            if ardisik_basari >= 10:
                print(f"\n*** Ajan yolu tamamen öğrendi! Eğitim {bolum + 1}. bölümde erken bitiriliyor. ***")
                break

        print("\n--- EĞİTİM BİTTİ ---")
        
        # ---------------------------------------------------------
        # TOTALLY GREEDY TEST AŞAMASI
        # ---------------------------------------------------------
        print("\n--- TEST BAŞLIYOR (Totally Greedy: Epsilon = 0.0) ---")
        
        test_oyun_sayisi = 5
        basari_sayisi = 0
        
        # Epsilon'u 0 yaparak ajanı tamamen açgözlü (totally greedy) yapıyoruz.
        # Artık asla rastgele hareket etmeyecek, sadece en yüksek Q değerini seçecek.
        test_epsilon = 0.0 
        
        for test in range(test_oyun_sayisi):
            state, _ = self.environment.reset()
            done = False
            adim_sayisi = 0
            
            while not done:
                # Ajan epsilon 0 olduğu için öğrendiği en optimal eylemi seçecektir
                action = self.agent.eylem_sec(state, test_epsilon) 
                state, reward, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                adim_sayisi += 1
                
                if terminated:
                    if state == self.durum_sayisi - 1: 
                        print(f"Test {test+1}: Hedefe başarıyla ulaşıldı! (Adım: {adim_sayisi})")
                        basari_sayisi += 1
                    else:
                        print(f"Test {test+1}: Deliğe düşüldü.")

        print(f"\nSonuç: {test_oyun_sayisi} oyunun {basari_sayisi} tanesinde hedefe ulaşıldı.")
        
        

import FrozenlakeWrapper
import QLearning

class agentBasedQLearning:
    
    def __init__(self,environment,episode,w_delik,w_hedef):
        self.w_delik = w_delik
        self.w_hedef = w_hedef
        self.environment = PBRSFrozenLakeWrapper(env=environment,w_delik=self.w_delik,w_hedef=self.w_hedef) 
        self.agent = QLearning.QLearningAgent(self.environment.observation_space.n, self.environment.action_space.n)
        self.episode = episode
        self.durum_sayisi = self.environment.observation_space.n
        
    
    def trainAgent(self):
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_azalma = 0.995
        print("--- EĞİTİM BAŞLIYOR (Epsilon-Greedy) ---")
        ardisik_basari = 0
        for bolum in range(self.episode):
            state, _ = self.environment.reset()
            done = False
            
            while not done:
                # 1. Eylem seç
                action = self.agent.eylem_sec(state, epsilon)
                
                # 2. Ortamda adımı at (PBRS Wrapper devredeyse şekillendirilmiş ödül gelir)
                next_state, reward, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                
                # 3. Ajanı eğit (Tabloyu anında güncelle)
                self.agent.ogren(state, action, reward, next_state, done)
                
                # 4. Durumu güncelle
                state = next_state
                if terminated:
                    if state == self.durum_sayisi - 1: # Hedefe ulaştı
                        ardisik_basari += 1
                    else: # Deliğe düştü
                        ardisik_basari = 0
                
            # Bölüm sonu Epsilon'u düşür (Ajan gittikçe daha az rastgele hareket etsin)
            if epsilon > epsilon_min:
                epsilon *= epsilon_azalma
            
            if ardisik_basari >= 10:
                print(f"\n*** Harika! Ajan Q-Tablosunu tamamen çözdü. Eğitim {bolum + 1}. bölümde erken bitiriliyor. ***")
                break

        print("\n--- EĞİTİM BİTTİ ---")
        
        # ---------------------------------------------------------
        # TOTALLY GREEDY TEST AŞAMASI
        # ---------------------------------------------------------
        print("\n--- TEST BAŞLIYOR (Totally Greedy: Epsilon = 0.0) ---")
        
        test_oyun_sayisi = 5
        basari_sayisi = 0
        test_epsilon = 0.0 # Keşif kapalı! Sadece en iyi bildiğini yapacak.
        
        for test in range(test_oyun_sayisi):
            state, _ = self.environment.reset()
            done = False
            adim_sayisi = 0
            
            while not done:
                # Totally Greedy eylem seçimi
                action = self.agent.eylem_sec(state, test_epsilon) 
                state, reward, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                adim_sayisi += 1
                
                if terminated:
                    if state == self.durum_sayisi - 1:
                        print(f"Test {test+1}: Hedefe başarıyla ulaşıldı! (Adım: {adim_sayisi})")
                        basari_sayisi += 1
                    else:
                        print(f"Test {test+1}: Deliğe düşüldü.")

        print(f"\nSonuç: {test_oyun_sayisi} oyunun {basari_sayisi} tanesinde hedefe ulaşıldı.")
        

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym

# Daha önce yazdığımız PBRSFrozenLakeWrapper sınıfının burada 
# import edildiğini veya tanımlandığını varsayıyoruz.
# from pbrs_wrapper import PBRSFrozenLakeWrapper 

# ---------------------------------------------------------
# 1. SİNİR AĞI SINIFI (Q-Network)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. DENEYİM HAFIZASI SINIFI (Replay Buffer)
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, kapasite=10000):
        self.hafiza = deque(maxlen=kapasite)

    def ekle(self, durum, eylem, odul, sonraki_durum, bitti_mi):
        self.hafiza.append((durum, eylem, odul, sonraki_durum, bitti_mi))

    def orneklem_al(self, batch_size):
        return random.sample(self.hafiza, batch_size)

    def __len__(self):
        return len(self.hafiza)

# ---------------------------------------------------------
# 3. DQN AJANI SINIFI
# ---------------------------------------------------------
class DQNAgent:
    def __init__(self, durum_sayisi, eylem_sayisi, lr=0.001, gamma=0.99):
        self.durum_sayisi = durum_sayisi
        self.eylem_sayisi = eylem_sayisi
        self.gamma = gamma
        self.batch_size = 64
        
        # Ana Ağ ve Hedef Ağ (Target Network)
        self.q_agi = QNetwork(durum_sayisi, eylem_sayisi)
        self.hedef_ag = QNetwork(durum_sayisi, eylem_sayisi)
        self.hedef_ag.load_state_dict(self.q_agi.state_dict())
        self.hedef_ag.eval() # Hedef ağ sadece değerlendirme için kullanılır
        
        self.optimizer = optim.Adam(self.q_agi.parameters(), lr=lr)
        self.hafiza = ReplayBuffer()

    def state_to_tensor(self, state):
        # Ayrık durumu (örn: 14) One-Hot vektöre çevirir: [0,0,..,1,0]
        one_hot = np.zeros(self.durum_sayisi)
        one_hot[state] = 1.0
        return torch.FloatTensor(one_hot).unsqueeze(0)

    def eylem_sec(self, state, epsilon):
        # Epsilon-Greedy Politikası
        if random.random() < epsilon:
            return random.randint(0, self.eylem_sayisi - 1) # Keşif (Rastgele)
        else:
            # Sömürü (Tamamen Açgözlü / Totally Greedy)
            state_tensor = self.state_to_tensor(state)
            with torch.no_grad():
                q_degerleri = self.q_agi(state_tensor)
            return torch.argmax(q_degerleri).item()

    def ogren(self):
        if len(self.hafiza) < self.batch_size:
            return # Yeterli deneyim yoksa öğrenme yapma

        batch = self.hafiza.orneklem_al(self.batch_size)
        durumlar, eylemler, oduller, sonraki_durumlar, bitisler = zip(*batch)

        # Batch'leri PyTorch tensörlerine çevirme
        durumlar_tensor = torch.cat([self.state_to_tensor(s) for s in durumlar])
        sonraki_durumlar_tensor = torch.cat([self.state_to_tensor(s) for s in sonraki_durumlar])
        eylemler_tensor = torch.LongTensor(eylemler).unsqueeze(1)
        oduller_tensor = torch.FloatTensor(oduller).unsqueeze(1)
        bitisler_tensor = torch.FloatTensor(bitisler).unsqueeze(1)

        # Q(s, a) hesaplama
        q_mevcut = self.q_agi(durumlar_tensor).gather(1, eylemler_tensor)

        # max Q(s', a') hesaplama (Hedef Ağ üzerinden)
        with torch.no_grad():
            q_sonraki = self.hedef_ag(sonraki_durumlar_tensor).max(1)[0].unsqueeze(1)
            # Eğer oyun bittiyse (delik veya hedef), sonraki Q değeri 0 olmalıdır
            q_hedef = oduller_tensor + (self.gamma * q_sonraki * (1 - bitisler_tensor))

        # Kayıp (Loss) hesabı ve ağırlıkların güncellenmesi
        loss = nn.MSELoss()(q_mevcut, q_hedef)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def hedef_agi_guncelle(self):
        self.hedef_ag.load_state_dict(self.q_agi.state_dict())
        
        
import numpy as np
import random
import gymnasium as gym

# Daha önce yazdığımız PBRSFrozenLakeWrapper sınıfının burada 
# import edildiğini veya tanımlandığını varsayıyoruz.
# from pbrs_wrapper import PBRSFrozenLakeWrapper 

# ---------------------------------------------------------
# 1. TABLOLU Q-ÖĞRENME AJANI SINIFI
# ---------------------------------------------------------
class QLearningAgent:
    def __init__(self, durum_sayisi, eylem_sayisi, lr=0.1, gamma=0.99):
        self.durum_sayisi = durum_sayisi
        self.eylem_sayisi = eylem_sayisi
        self.lr = lr       # Öğrenme katsayısı (Alpha)
        self.gamma = gamma # İndirim faktörü

        # Ajanın Beyni: Q-Tablosu (Başlangıçta her şey sıfır)
        # Boyutlar: [Durum Sayısı x Eylem Sayısı] (Örn: 16x4)
        self.q_tablosu = np.zeros((durum_sayisi, eylem_sayisi))

    def eylem_sec(self, state, epsilon):
        # Epsilon-Greedy Politikası
        if random.uniform(0, 1) < epsilon:
            # Keşif (Exploration): Rastgele bir eylem seç
            return random.randint(0, self.eylem_sayisi - 1)
        else:
            # Sömürü (Exploitation): Tablodaki en yüksek Q değerine sahip eylemi seç
            return np.argmax(self.q_tablosu[state, :])

    def ogren(self, state, action, reward, next_state, done):
        # Bellman Denklemi ile Q-Tablosunun Güncellenmesi
        
        # Oyun bittiyse (delik veya hedef) sonraki durumun tahmini değeri 0 olur
        if done:
            max_gelecek_q = 0
        else:
            max_gelecek_q = np.max(self.q_tablosu[next_state, :])

        # Mevcut Q değeri
        mevcut_q = self.q_tablosu[state, action]
        
        # Yeni Q değeri (Öğrenme Katsayısı ile harmanlanmış)
        yeni_q = mevcut_q + self.lr * (reward + self.gamma * max_gelecek_q - mevcut_q)
        
        # Tabloyu güncelle
        self.q_tablosu[state, action] = yeni_q


