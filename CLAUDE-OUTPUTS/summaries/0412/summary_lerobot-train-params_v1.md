# lerobot-train 파라미터 정리

> **요약 (2-3줄):** `lerobot-train`의 파라미터는 훈련 공통(dataset/training), ACT 정책, Diffusion 정책 세 범주로 나뉜다. ACT는 기본적으로 ResNet18(CNN) backbone을 사용하며 ViT가 아니다. Diffusion은 UNet CNN 기반으로 별도의 노이즈 스케줄러 파라미터가 추가된다.

---

## 1. 공통 훈련 파라미터 (`--[파라미터명]=값`)

### Dataset

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `dataset.repo_id` | 필수 | HuggingFace 데이터셋 ID (e.g. `JungSeong2/AIC`) |
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

## 5. ACT vs Diffusion 비교

| 항목 | ACT | Diffusion |
|------|-----|-----------|
| backbone | ResNet (CNN) | ResNet (CNN) |
| 훈련 속도 | 빠름 | 보통 |
| Inference 속도 | 빠름 (단일 forward) | 느림 (denoising 반복) |
| 메모리 | 보통 | 많음 (UNet) |
| 특징 | Action chunking, VAE | 확률 분포 모델링 |
| 기본 LR | `1e-5` | `1e-4` |
| 빠른 Inference | `temporal_ensemble_coeff` | `DDIM` + 적은 steps |
