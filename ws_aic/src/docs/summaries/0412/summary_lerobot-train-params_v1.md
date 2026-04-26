# lerobot-train 파라미터 정리

> **요약 (2-3줄):** `lerobot-train`의 파라미터는 훈련 공통(dataset/training), 정책별 파라미터로 나뉜다. 지원하는 `policy.type`은 14종이며 모방학습(ACT, Diffusion, VQ-BeT), VLA(pi0, SmolVLA, xVLA), 모델 기반 RL(TD-MPC), 온라인 RL(SAC) 등 크게 4가지 패러다임으로 분류된다. AIC 태스크에는 **ACT** 가 기본 선택이며 데이터가 충분하면 **Diffusion** 또는 **pi0** 계열 고려 가능.

---

## 0. policy.type 전체 종류 및 특징

`lerobot/policies/factory.py` 기준 지원 타입 14종.

### 패러다임별 분류

#### 모방학습 (Imitation Learning) — 오프라인 데이터셋 기반

| type | 풀네임 | 특징 | AIC 적합도 |
|------|--------|------|-----------|
| **`act`** | Action Chunking with Transformers | ResNet18 + Transformer + VAE. chunk 단위 action 예측으로 떨림 감소. 빠른 inference. | ★★★ **기본 선택** |
| **`diffusion`** | Diffusion Policy | UNet + DDPM/DDIM. 복잡한 멀티모달 행동 분포 표현 가능. inference 느림. | ★★ 데이터 많을 때 |
| **`vqbet`** | Vector Quantized BeT | VQ-VAE로 action을 이산화 후 Transformer로 예측. 연속 action 공간에 약간 불리. | ★ |

#### VLA (Vision-Language-Action) — 대형 사전학습 모델 기반

| type | 풀네임 | 특징 | AIC 적합도 |
|------|--------|------|-----------|
| **`pi0`** | π₀ | PaliGemma(3B) 기반 VLA. 언어 지시 + 비전 + action 통합. VRAM 24GB+ 필요. | ★★ VRAM 충분 시 |
| **`pi05`** | π₀.5 | π₀ 업그레이드. 더 강한 범용성. | ★★ |
| **`pi0_fast`** | π₀ Fast | π₀의 inference 최적화 버전. | ★★ |
| **`smolvla`** | SmolVLA | SmolLM 기반 소형 VLA. pi0보다 VRAM 적게 필요. | ★★ |
| **`xvla`** | xVLA | 또 다른 VLA 계열. | ★ |
| **`groot`** | GR00T | NVIDIA 제작 범용 로봇 학습 모델. VRAM 요구량 높음. | ★ (버그 있음) |

#### 모델 기반 RL (Model-Based RL)

| type | 풀네임 | 특징 | AIC 적합도 |
|------|--------|------|-----------|
| **`tdmpc`** | TD-MPC | 환경 모델 학습 + MPC로 계획. 오프라인 데이터로도 사용 가능. | ★ |

#### 온라인 RL (Online Reinforcement Learning)

| type | 풀네임 | 특징 | AIC 적합도 |
|------|--------|------|-----------|
| **`sac`** | Soft Actor-Critic | 환경과 실시간 상호작용 필요. 오프라인 데이터셋만으로는 사용 불가. | ✗ (시뮬 연동 필요) |

#### 기타 (내부/보조 모델)

| type | 설명 |
|------|------|
| `sarm` | Reward Model (보상 함수 학습용) |
| `reward_classifier` | 보상 분류기 |
| `wall_x` | 특수 실험용 정책 |

---

### AIC 태스크 기준 선택 가이드

```
데이터 수십 에피소드   → act  (빠르고 안정적)
데이터 수백 에피소드   → diffusion 또는 act
VRAM 24GB+, 고성능    → pi0 / smolvla (언어 지시 활용 가능)
온라인 RL 원할 때      → sac (Gazebo 환경 직접 연동 필요, 별도 구현 필요)
```

---

### 각 policy.type의 추론 출력 (Action Space)

**policy.type은 아키텍처만 결정하고, 출력 차원은 학습에 사용된 데이터셋이 결정합니다.**

모든 오프라인 모방학습 계열(ACT, Diffusion, VQ-BeT, pi0, SmolVLA 등)은 학습 데이터셋의 action 공간을 그대로 출력합니다.

| 녹화 시 설정 | 출력 action | 차원 | 예시 |
|-------------|-------------|------|------|
| `--robot.teleop_target_mode=cartesian` | Cartesian 속도 명령 | **6** | `[linear.x, y, z, angular.x, y, z]` |
| `--robot.teleop_target_mode=joint` | 관절 속도 명령 | **6** | `[shoulder_pan, lift, elbow, wrist_1, 2, 3]` |

> **RunACT.py에서 "(1, 7)" 언급이 있는 이유:** `get_stat("action.mean", (1, -1))`에서 `-1`은 실제 차원을 safetensors에서 자동으로 읽어오는 것. 저장된 데이터에 따라 6 또는 다른 값이 될 수 있음. 현재 AIC Cartesian 수집 기준으로는 **6차원**.

**policy.type별로 달라지는 것:**

| 항목 | 모방학습 계열 | VLA 계열 (pi0, SmolVLA) | SAC |
|------|--------------|------------------------|-----|
| **입력** | 이미지 + 로봇 상태 | 이미지 + 로봇 상태 + **언어 지시** | 환경 상태 |
| **출력** | 데이터셋 action 그대로 | 데이터셋 action 그대로 | 환경 action 공간 |
| **데이터 필요** | 오프라인 데이터셋 | 오프라인 데이터셋 | **실시간 환경 상호작용** |
| **아키텍처** | 각기 다름 | 대형 LLM/VLM backbone | Actor-Critic 네트워크 |

> **SAC만 예외**: 데이터셋 기반이 아닌 온라인 RL이라 action space를 환경에서 직접 정의. lerobot-record 데이터와 무관.

---

## 1. 공통 훈련 파라미터 (`--[파라미터명]=값`)

### Dataset

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `dataset.repo_id` | 필수 | HuggingFace 데이터셋 ID (e.g. `aic-sejong-team/AIC`) |
| `dataset.root` | `None` | 로컬 데이터셋 경로 (HF 대신 사용) |

### 훈련 설정

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `steps` | `100_000` | 전체 훈련 스텝 수 |
| `batch_size` | `8` | 배치 크기 |
| `num_workers` | `4` | DataLoader 워커 수 |
| `seed` | `1000` | 재현성 시드 |
| `resume` | `false` | 이전 체크포인트에서 재개 |
| `cudnn_deterministic` | `false` | 결정론적 cuDNN (속도 10~20% 감소) |

### 저장 / 로그

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `output_dir` | 자동생성 | 출력 저장 경로 |
| `job_name` | 자동생성 | 실행 이름 |
| `save_checkpoint` | `true` | 체크포인트 저장 여부 |
| `save_freq` | `20_000` | 체크포인트 저장 주기 (스텝) |
| `log_freq` | `200` | 로그 출력 주기 (스텝) |
| `eval_freq` | `20_000` | 평가 주기 (스텝) |

### WandB

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `wandb.enable` | `false` | WandB 활성화 |
| `wandb.project` | `lerobot` | 프로젝트 이름 |
| `wandb.entity` | `None` | 팀/유저 이름 |
| `wandb.run_id` | `None` | 실행 ID (재개 시) |

### Policy 공통

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `policy.type` | 필수 | 정책 종류: `act`, `diffusion`, `tdmpc`, `pi0` 등 |
| `policy.device` | `cpu` | 실행 장치: `cuda`, `cpu`, `mps` |
| `policy.repo_id` | `None` | HF에 업로드할 모델 ID |
| `policy.push_to_hub` | `false` | 훈련 후 HF Hub에 자동 업로드 |

---

## 2. ACT 파라미터 (`--policy.type=act`)

> ACT의 vision backbone은 기본값이 **ResNet18(CNN)** — ViT가 아님.

### 입출력 구조

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `policy.n_obs_steps` | `1` | 입력으로 사용할 과거 observation 스텝 수 (현재 1만 지원) |
| `policy.chunk_size` | `100` | 한 번에 예측하는 action 시퀀스 길이 |
| `policy.n_action_steps` | `100` | 한 번 호출로 실제 실행하는 action 수 (`≤ chunk_size`) |

### Vision Backbone (CNN)

| 파라미터 | 기본값 | 선택지 | 설명 |
|----------|--------|--------|------|
| `policy.vision_backbone` | `resnet18` | `resnet18`, `resnet34`, `resnet50` | 비전 인코더 |
| `policy.pretrained_backbone_weights` | `ResNet18_Weights.IMAGENET1K_V1` | `null`, `ResNet18_Weights.IMAGENET1K_V1` | 사전학습 가중치 (`null`로 설정 시 빠름) |
| `policy.replace_final_stride_with_dilation` | `false` | `true`/`false` | 마지막 stride를 dilated conv로 교체 |

### Transformer 구조

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `policy.dim_model` | `512` | Transformer hidden dimension |
| `policy.n_heads` | `8` | Multi-head attention 헤드 수 |
| `policy.dim_feedforward` | `3200` | FFN hidden dimension |
| `policy.n_encoder_layers` | `4` | Encoder 레이어 수 |
| `policy.n_decoder_layers` | `1` | Decoder 레이어 수 |
| `policy.feedforward_activation` | `relu` | FFN 활성화 함수 |
| `policy.pre_norm` | `false` | Pre-norm 사용 여부 |

### VAE (변분 오토인코더)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `policy.use_vae` | `true` | VAE 목적함수 사용 여부 |
| `policy.latent_dim` | `32` | VAE latent 차원 |
| `policy.n_vae_encoder_layers` | `4` | VAE 인코더 레이어 수 |
| `policy.kl_weight` | `10.0` | KL divergence 가중치 (loss = recon + kl_weight × kld) |

### 추론 / 훈련

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `policy.temporal_ensemble_coeff` | `null` | Temporal ensembling 계수 (사용 시 `n_action_steps=1` 강제) |
| `policy.dropout` | `0.1` | Transformer dropout 비율 |
| `policy.optimizer_lr` | `1e-5` | 학습률 |
| `policy.optimizer_lr_backbone` | `1e-5` | backbone 학습률 |
| `policy.optimizer_weight_decay` | `1e-4` | Weight decay |

---

## 3. Diffusion Policy 파라미터 (`--policy.type=diffusion`)

### 입출력 구조

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `policy.n_obs_steps` | `2` | 입력 observation 스텝 수 |
| `policy.horizon` | `16` | 예측 action 시퀀스 길이 (UNet 다운샘플링 배수여야 함) |
| `policy.n_action_steps` | `8` | 실제 실행 action 수 |

### Vision Backbone (CNN)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `policy.vision_backbone` | `resnet18` | 비전 인코더 (ResNet 계열만 지원) |
| `policy.pretrained_backbone_weights` | `null` | 사전학습 가중치 (기본 없음) |
| `policy.resize_shape` | `null` | 이미지 전처리 resize `[H, W]` |
| `policy.crop_ratio` | `1.0` | crop 비율 (1.0 = no crop) |
| `policy.crop_is_random` | `true` | 훈련 시 랜덤 crop 여부 |
| `policy.use_group_norm` | `true` | BatchNorm → GroupNorm 교체 |
| `policy.spatial_softmax_num_keypoints` | `32` | SpatialSoftmax keypoint 수 |

### UNet (Diffusion 모델 본체)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `policy.down_dims` | `(512, 1024, 2048)` | 각 다운샘플 스테이지의 채널 수 (길이 = 다운샘플 단계 수) |
| `policy.kernel_size` | `5` | Conv 커널 크기 |
| `policy.n_groups` | `8` | GroupNorm 그룹 수 |
| `policy.diffusion_step_embed_dim` | `128` | Diffusion timestep 임베딩 차원 |
| `policy.use_film_scale_modulation` | `true` | FiLM scale modulation 사용 |

### 노이즈 스케줄러

| 파라미터 | 기본값 | 선택지 | 설명 |
|----------|--------|--------|------|
| `policy.noise_scheduler_type` | `DDPM` | `DDPM`, `DDIM` | 노이즈 스케줄러 종류 |
| `policy.num_train_timesteps` | `100` | | Forward diffusion 스텝 수 |
| `policy.num_inference_steps` | `null` (=train과 동일) | | Inference 시 denoising 스텝 |
| `policy.beta_schedule` | `squaredcos_cap_v2` | | Beta 스케줄 |
| `policy.beta_start` | `0.0001` | | Beta 시작값 |
| `policy.beta_end` | `0.02` | | Beta 끝값 |
| `policy.prediction_type` | `epsilon` | `epsilon`, `sample` | 예측 대상 유형 |
| `policy.clip_sample` | `true` | | 샘플 클리핑 여부 |
| `policy.clip_sample_range` | `1.0` | | 클리핑 범위 `[-1, 1]` |

### 옵티마이저 / 스케줄러

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `policy.optimizer_lr` | `1e-4` | 학습률 |
| `policy.optimizer_weight_decay` | `1e-6` | Weight decay |
| `policy.scheduler_name` | `cosine` | LR 스케줄러 |
| `policy.scheduler_warmup_steps` | `500` | Warmup 스텝 수 |
| `policy.compile_model` | `false` | `torch.compile` 사용 여부 |

---

## 4. 빠른 테스트 예시 (CNN, 빠른 실험용)

```bash
# ACT + ResNet18 (pretrained 없음, 빠른 테스트)
pixi run lerobot-train \
  --dataset.repo_id=JungSeong2/AIC \
  --policy.type=act \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=null \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.dim_model=256 \
  --policy.n_encoder_layers=2 \
  --steps=10000 \
  --batch_size=16 \
  --save_freq=5000 \
  --output_dir=/home/vsc/LLM_TUNE/AIC_Sejong/data/outputs/train/act_cnn_test \
  --job_name=act_cnn_test \
  --policy.device=cuda \
  --wandb.enable=false
```

```bash
# Diffusion (DDIM으로 빠른 inference)
pixi run lerobot-train \
  --dataset.repo_id=JungSeong2/AIC \
  --policy.type=diffusion \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=10 \
  --policy.down_dims="(256,512,1024)" \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --steps=10000 \
  --batch_size=16 \
  --output_dir=/home/vsc/LLM_TUNE/AIC_Sejong/data/outputs/train/diffusion_test \
  --job_name=diffusion_test \
  --policy.device=cuda \
  --wandb.enable=false
```

---

## 5. 주요 정책 비교

| 항목 | ACT | Diffusion | pi0 / SmolVLA |
|------|-----|-----------|---------------|
| **패러다임** | 모방학습 | 모방학습 | VLA (모방학습) |
| **backbone** | ResNet18 (CNN) | ResNet18 (CNN) | PaliGemma / SmolLM |
| **훈련 속도** | 빠름 | 보통 | 느림 |
| **Inference 속도** | 빠름 (단일 forward) | 느림 (denoising 반복) | 보통~느림 |
| **VRAM** | ~8GB | ~12GB | 24GB+ / 16GB+ |
| **필요 데이터** | 수십 에피소드~ | 수백 에피소드~ | 소량도 가능 (사전학습 덕분) |
| **핵심 특징** | Action chunking + VAE | 멀티모달 분포 표현 | 언어 지시 가능, 범용성 높음 |
| **기본 LR** | `1e-5` | `1e-4` | 모델마다 다름 |
| **빠른 Inference 옵션** | `temporal_ensemble_coeff` | `DDIM` + 적은 steps | `pi0_fast` |
| **AIC 추천도** | ★★★ | ★★ | ★★ (VRAM 여유 시) |
