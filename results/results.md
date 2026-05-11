# Inference-Time Temporal Smoothing for Vision-Language-Action Models via Learned Kalman Filter

> **VLA-JEPA의 stateless encoder가 유발하는 token-level 고주파 변동을 Learned LDS Kalman Filter로 억제하여, 모델 재학습 없이 inference-time에 행동 안정성을 개선한다.**

---

## 1. Introduction

### 1.1 Background: VLA-JEPA Architecture

VLA-JEPA는 Qwen3-VL-2B (QwenVL)과 DiT(Diffusion Transformer) 기반 flow-matching action head를 결합한 Vision-Language-Action 모델이다. 학습 시에는 V-JEPA2 ViT-L을 world model teacher로 사용하여 action-conditioned 미래 시각 상태 예측을 통해 지식을 증류하지만, **inference 시에는 QwenVL + DiT만 사용**한다.

```
[학습 시]
Images + Language
       ↓
    QwenVL (Qwen3-VL-2B)
       ↓
embodied_action_tokens  [B, 32, 2048]
       ├──→ JEPA predictor → future visual state prediction (teacher signal)
       └──→ DiT (Flow-matching) → action chunk [B, 7, action_dim]

[Inference 시]
Images + Language → QwenVL → embodied_action_tokens → DiT → action chunk
                    (V-JEPA2 미사용)
```

Action chunk size = 7: 7 스텝마다 1회 inference를 수행하며, 각 호출에서 7개의 미래 액션을 생성한다.

### 1.2 Problem: Temporal Inconsistency of Stateless Encoding

QwenVL은 매 inference call마다 **현재 프레임만 독립적으로 인코딩**한다 (no temporal context). 실제 로봇 환경에서 연속된 프레임은 시각적으로 거의 동일하지만, 트랜스포머의 비선형 어텐션 메커니즘은 미세한 픽셀 변화에도 민감하게 반응하여 `embodied_action_tokens`를 timestep 간에 불필요하게 변동시킨다.

```
frame(t)   → QwenVL → tokens(t):   [0.82, -0.31, 0.67, ...]
frame(t+1) → QwenVL → tokens(t+1): [0.79, -0.35, 0.71, ...]  ← ~1mm 이동에도 변동
```

이 고주파 변동(high-frequency fluctuation)이 DiT의 conditioning으로 전달되어 **action chunk 간 불일치**를 유발한다. 직접 action에 smoothing을 적용하면 7D motor space에서 물리적으로 inconsistent한 중간 자세가 생성되는 문제가 있어, 더 높은 수준의 의미 공간(semantic space)에서의 smoothing이 필요하다.

### 1.3 Key Insight

`embodied_action_tokens`는 JEPA 학습 목표(미래 시각 상태 예측)를 통해 **world model 정보가 압축된 semantic bottleneck**이다. 이 공간에서 smoothing하면:
- DiT가 안정된 "의도(intent)"로부터 일관된 action chunk를 생성
- Motor space averaging과 달리 물리적 inconsistency 없음
- 학습된 dynamics 모델을 활용하여 단순 이동 평균 대비 우위

---

## 2. Method

### 2.1 Offline Learning: Learned LDS

데모 데이터로부터 1회 offline 학습을 수행한다.

**Step 1 — Token Extraction**

데모 시퀀스 N개에서 각 timestep의 `embodied_action_tokens`를 추출하고 mean-pooling하여 feature 시퀀스를 구성한다:

$$y_t = \frac{1}{32} \sum_{i=1}^{32} \text{token}_i^{(t)} \in \mathbb{R}^{2048}$$

**Step 2 — PCA Encoder**

Truncated SVD로 상위 64개 주성분을 추출하여 인코더 E를 학습:

$$E \in \mathbb{R}^{64 \times 2048}, \quad z_t = E y_t \in \mathbb{R}^{64}$$

**Step 3 — AR(1) Transition Matrix**

연속된 latent 쌍 $(z_{t-1}, z_t)$에 대해 최소제곱법으로 전이행렬 A를 학습:

$$A = \arg\min_{A} \sum_{t} \| z_t - A z_{t-1} \|^2, \quad A \in \mathbb{R}^{64 \times 64}$$

### 2.2 Online Inference: Kalman Filter

매 inference call마다 다음 KF predict-update를 수행한다.

| 기호 | 의미 | 값 |
|------|------|-----|
| $Q$ | process noise covariance | $0.1 \cdot I_{64}$ |
| $R$ | observation noise covariance | $5.0 \cdot I_{64}$ |
| $P_0$ | 초기 state covariance | $I_{64}$ (uninformed prior) |

**Predict:**

$$\hat{z}_t^- = A z_{t-1}, \qquad P_t^- = A P_{t-1} A^\top + Q$$

**Update:**

$$K_t = P_t^- (P_t^- + R)^{-1}$$

$$z_t = \hat{z}_t^- + K_t (z_t^{\text{obs}} - \hat{z}_t^-), \qquad P_t = (I - K_t) P_t^-$$

**Correction 적용:**

```
tokens [B, 32, 2048]  →  mean-pool  →  y_obs [B, 2048]
    → PCA encode  →  z_obs [B, 64]
    → KF predict + update  →  z_filtered [B, 64]
    → PCA decode  →  y_filtered [B, 2048]
correction = y_filtered - y_obs
corrected_tokens = tokens + correction.unsqueeze(1)  [B, 32, 2048]
    → DiT → action chunk [B, 7, action_dim]
```

### 2.3 Baseline Comparison: EMA

$$y_t^{\text{filtered}} = \alpha \cdot y_t + (1 - \alpha) \cdot y_{t-1}^{\text{filtered}}$$

학습 없이 적용 가능. KF 대비 learned dynamics(A) 없이 단순 과거 관측의 가중 평균. KF > EMA이면 "학습된 dynamics 모델이 중요하다"는 근거.

### 2.4 Insertion Point

```python
# VLA_JEPA.py: predict_action()
embodied_action_tokens = last_hidden[...].view(B, -1, H)  # QwenVL 출력
# ← KF 삽입 (QwenVL 출력 직후, DiT 입력 직전)
if self._lds is not None:
    y_raw      = tokens_np.mean(axis=1)
    y_filtered = [self._kf_step(y_raw[b]) for b in range(B)]
    correction = torch.from_numpy(y_filtered - y_raw)
    embodied_action_tokens = embodied_action_tokens + correction.unsqueeze(1)
pred_actions = self.action_model.predict_action(embodied_action_tokens, state)
```

---

## 3. Experiments

### 3.1 Setup

| 항목 | 값 |
|------|-----|
| 모델 | VLA-JEPA (Qwen3-VL-2B + DiT) |
| 평가 방식 | 10 tasks × 50 trials = 500 episodes / run |
| 반복 횟수 | 3회 (seed = 7, 42, 123) |
| LDS latent dim | 64 |
| KF Q | 0.1 · I |
| KF R | 5.0 · I |
| GPU | A6000 (48GB) × 1 |

### 3.2 Results

#### LIBERO-Spatial

| Method | Seed 7 | Seed 42 | Seed 123 | Mean ± Std | Δ | p-value |
|--------|--------|---------|----------|------------|---|---------|
| Baseline | 94.6% | 95.2% | 95.8% | 95.20% ± 0.49% | — | — |
| EMA α=0.5 | 94.4% | 95.2% | 95.4% | 95.00% ± 0.43% | −0.20%p | 0.687 |
| EMA α=0.7 | 94.4% | 95.4% | 95.0% | 94.93% ± 0.41% | −0.27%p | 0.587 |
| **KF (ours)** | **96.8%** | **97.0%** | **97.6%** | **97.13% ± 0.34%** | **+1.93%p** | **0.010\*** |

#### LIBERO-Object

| Method | Seed 7 | Seed 42 | Seed 123 | Mean ± Std | Δ | p-value |
|--------|--------|---------|----------|------------|---|---------|
| Baseline | 100.0% | 99.8% | 100.0% | 99.93% ± 0.09% | — | — |
| EMA α=0.5 | 100.0% | 99.8% | 100.0% | 99.93% ± 0.09% | +0.00%p | 1.000 |
| EMA α=0.7 | 100.0% | 100.0% | 100.0% | 100.00% ± 0.00% | +0.07%p | 0.374 |
| **KF (ours)** | 99.8% | 99.8% | 100.0% | 99.87% ± 0.09% | −0.07%p | 0.519 |

#### LIBERO-Goal

| Method | Seed 7 | Seed 42 | Seed 123 | Mean ± Std | Δ | p-value |
|--------|--------|---------|----------|------------|---|---------|
| Baseline | 97.8% | 97.6% | 95.8% | 97.07% ± 0.90% | — | — |
| EMA α=0.5 | 98.0% | 96.8% | 97.0% | 97.27% ± 0.52% | +0.20%p | 0.799 |
| EMA α=0.7 | 98.0% | 96.4% | 96.6% | 97.00% ± 0.71% | −0.07%p | 0.938 |
| **KF (ours)** | 97.6% | 98.0% | 97.6% | **97.73% ± 0.19%** | **+0.67%p** | 0.363 |

#### LIBERO-Long (libero_10)

| Method | Seed 7 | Seed 42 | Seed 123 | Mean ± Std | Δ | p-value |
|--------|--------|---------|----------|------------|---|---------|
| Baseline | 94.8% | 95.4% | 95.8% | 95.33% ± 0.41% | — | — |
| **KF (ours)** | **95.4%** | 진행 중 | 미시작 | — | — | — |

### 3.3 Cross-Suite Summary

| Suite | Baseline | KF | Δ | p-value | 비고 |
|-------|----------|----|---|---------|------|
| LIBERO-Spatial | 95.20% | 97.13% | **+1.93%p** | **0.010\*** | 유의함 |
| LIBERO-Object | 99.93% | 99.87% | −0.07%p | 0.519 | Ceiling effect |
| LIBERO-Goal | 97.07% | 97.73% | +0.67%p | 0.363 | std 0.90→0.19% |
| LIBERO-Long | 95.33% | TBD | TBD | TBD | 진행 중 |

---

## 4. Analysis

### 4.1 Why KF > EMA

| | EMA | KF |
|--|-----|----|
| Gain 결정 | 고정 α | $K_t = P_t^-(P_t^- + R)^{-1}$ (적응적) |
| Dynamics 활용 | 없음 | A로 다음 상태 예측 |
| 에피소드 초반 | α 고정 | P=I → K 크게 → 관측 빠르게 반영 |
| 에피소드 중반 | α 고정 | P 수렴 → K 작게 → 예측 신뢰 |

EMA는 이미 노이즈가 있는 과거 관측의 단순 평균으로 유효 정보를 희석시키는 반면, KF는 learned dynamics로 예측한 위치와 관측을 uncertainty 기반으로 합성한다. LIBERO-Spatial에서 EMA가 오히려 소폭 하락(−0.2~0.3%p)하는 반면 KF는 +1.93%p를 달성한 것이 이를 뒷받침한다.

### 4.2 Variance Reduction Effect

통계적 유의성이 낮은 suite에서도 KF는 일관되게 분산을 감소시킨다:

| Suite | Baseline std | KF std | 감소율 |
|-------|------------|--------|--------|
| LIBERO-Spatial | 0.49% | 0.34% | −31% |
| LIBERO-Goal | 0.90% | 0.19% | **−79%** |

분산 감소는 행동 안정성 향상의 직접적 지표다.

### 4.3 Ceiling Effect

Baseline이 ~100%에 가까운 LIBERO-Object에서는 개선 여지가 없어 어떤 방법도 통계적으로 유의한 차이를 보이지 못한다. KF의 효과는 **Baseline이 ~95% 수준일 때 가장 뚜렷**하다.

### 4.4 Noise Interpretation

전통적 KF의 noise(GPS 센서 오차 등)와 달리, 여기서의 "noise"는 물리적 랜덤 프로세스가 아닌 **트랜스포머의 비선형 민감성**에서 비롯된다. 시뮬레이션 환경조차 결정론적임에도 불구하고 QwenVL의 독립적 per-frame 인코딩이 token 변동을 유발한다. Q=0.1, R=5.0은 물리적으로 calibrated된 값이 아니라 smoothing 강도를 제어하는 hyperparameter로 해석하는 것이 정확하다.

---

## 5. Conclusion

**핵심 기여:**
1. **Training-free improvement**: 오프라인 LDS 학습(수 초)만으로 plug-in 적용
2. **통계적으로 유의한 성능 향상**: LIBERO-Spatial +1.93%p, p=0.010
3. **분산 감소**: LIBERO-Goal std −79% 등 행동 안정성 향상
4. **EMA 대비 KF 우위 실증**: learned dynamics 모델의 필요성 입증

**한계 및 향후 연구:**
- Ceiling effect: ~100% 수준의 suite에서는 측정 불가
- Q/R isotropy 가정: EM 알고리즘으로 Q, R을 데이터에서 학습 시 개선 가능
- 선형 dynamics 가정: SSM/Mamba 등 비선형 모델로의 확장 가능
- JEPA 직접 활용: inference-time JEPA predictor를 사용한 MPC 방식이 다음 연구 방향

---

## Appendix

### A. File Structure

```
vjepa2/
├── src/models/kf/learned_lds.py
└── experiments/vla_jepa/
    ├── extract_tokens.py
    └── train_lds_tokens.py

VLA-JEPA/
├── run_all_suites.sh
├── run_libero_pro.sh
├── starVLA/model/framework/VLA_JEPA.py
├── deployment/model_server/server_policy.py
└── results/
    ├── libero_spatial/{baseline,kf,EMA_0p5,EMA_0p7}_run{1,2,3}/
    ├── libero_object/{baseline,kf,EMA_0p5,EMA_0p7}_run{1,2,3}/
    ├── libero_goal/{baseline,kf,EMA_0p5,EMA_0p7}_run{1,2,3}/
    └── libero_10/{baseline,kf}_run{1,2,3}/
```

### B. Hyperparameters

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| latent_dim | 64 | PCA 압축 차원 |
| kf_q | 0.1 | Process noise (smoothing 강도 조절) |
| kf_r | 5.0 | Observation noise (smoothing 강도 조절) |
| P_init | I | 에피소드 초기 불확실성 |
| EMA α | 0.5, 0.7 | 현재 관측 반영 비율 |

### C. Data Split

| 데이터 | 용도 |
|--------|------|
| demo_0 ~ demo_39 (task당 40개) | LDS 학습 |
| demo_40 ~ demo_49 (task당 10개) | LDS 검증 |
| 평가 rollout (50 fixed init states × 10 tasks) | 성능 평가 |
