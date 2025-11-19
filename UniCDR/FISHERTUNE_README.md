# FisherTune for UniCDR

FisherTune을 UniCDR에 적용한 구현체입니다. Domain-Related Fisher Information을 활용하여 Cross-Domain Recommendation의 성능을 향상시킵니다.

## 주요 특징

1. **Domain-Related Fisher Information (DR-FIM)**: 도메인 간 파라미터 민감도를 측정하여 중요한 파라미터를 식별
2. **Shared/Specific Parameter Separation**: 공유 파라미터와 도메인 특화 파라미터에 다른 학습 전략 적용
3. **Progressive Parameter Selection**: 학습 진행에 따라 점진적으로 더 많은 파라미터를 튜닝
4. **Domain Perturbation**: Edge dropout, popularity weighting 등을 통한 robust한 FIM 추정
5. **Variational Inference**: 안정적인 Fisher 추정을 위한 변분 추론

## 설치

추가 의존성 없이 기존 UniCDR 환경에서 사용 가능합니다.

```bash
# 필요한 패키지
pip install torch numpy scipy pandas
```

## Quick Start

### 1. 기본 FisherTune 학습

```bash
python train_fishertune.py \
    --domains sport_cloth \
    --task dual-user-intra \
    --cuda \
    --use_fishertune \
    --fishertune_warmup 10 \
    --shared_lr 0.0001 \
    --specific_lr 0.001
```

### 2. 실험 스크립트 사용

```bash
# 모든 실험 실행
./run_experiments.sh all

# 특정 실험만 실행
./run_experiments.sh default
./run_experiments.sh conservative
./run_experiments.sh all_features

# 빠른 비교 (baseline + default + all_features)
./run_experiments.sh quick
```

### 3. 결과 분석

```bash
python analyze_results.py --log_dir logs --output results/report.md
```

## 주요 하이퍼파라미터

### Shared vs Specific Parameter 분리

```bash
# Shared parameters: 높은 δ (대부분 동결, 불변성 유지)
--shared_delta_min 0.5      # 최종 threshold (더 많은 파라미터 선택)
--shared_delta_max 0.95     # 초기 threshold (적은 파라미터 선택)
--shared_lr 0.0001          # 낮은 학습률
--shared_weight_decay 1e-3  # 강한 정규화

# Specific parameters: 낮은 δ (넓게 tuning 허용)
--specific_delta_min 0.1    # 최종 threshold
--specific_delta_max 0.7    # 초기 threshold
--specific_lr 0.001         # 높은 학습률
--specific_weight_decay 1e-5 # 약한 정규화
```

### Domain Perturbation

```bash
# Edge Dropout
--perturbation_type edge_dropout
--edge_dropout_rate 0.2

# Popularity-based weighting
--perturbation_type popularity
--popularity_alpha 0.5

# Noise injection
--perturbation_type noise
--perturbation_noise_std 0.1

# Combined (모든 전략 함께 사용)
--perturbation_type combined
```

### Variational Inference

```bash
# 기본 VI
--variational_tau 1.0       # Prior std
--variational_gamma 0.1     # KL weight

# Adaptive VI (자동 조절)
--use_adaptive_vi
--tau_decay 0.99
--gamma_growth 1.01
```

### Advanced Options

```bash
# Natural Gradient (파라미터 기하학 고려)
--use_natural_gradient
--fisher_damping 1e-4

# Fisher Regularization (중요 파라미터 보호)
--use_fisher_regularization
--fisher_reg_strength 0.1

# Adaptive Scheduler (학습 동태에 따른 threshold 조절)
--use_adaptive_scheduler
--scheduler_adaptation_rate 0.1
```

## 아키텍처

```
fishertune/
├── __init__.py                    # Module exports
├── fisher_info.py                 # FIM computation & DR-FIM
├── domain_perturbation.py         # Domain perturbation strategies
├── variational_fisher.py          # Variational inference
├── parameter_scheduler.py         # Progressive parameter selection
├── fishertune_optimizer.py        # Optimizer wrapper
└── fishertune_trainer.py          # Main trainer integration
```

## 성능 향상 전략

### 1. Conservative Shared (공유 파라미터 보수적 접근)
- **목표**: 도메인 불변 지식 보존
- **설정**: `shared_delta_min=0.7`, `shared_lr=0.00005`
- **효과**: 과적합 방지, 일반화 성능 향상

### 2. Aggressive Specific (도메인 특화 파라미터 적극적 튜닝)
- **목표**: 도메인 특화 지식 최대 활용
- **설정**: `specific_delta_min=0.05`, `specific_lr=0.002`
- **효과**: 도메인별 성능 최적화

### 3. Long Warmup (긴 워밍업)
- **목표**: pretrained model 부재로 인한 불안정성 해소
- **설정**: `fishertune_warmup=20`
- **효과**: 안정적인 FIM 추정 기반 확보

### 4. Frequent FIM Updates
- **목표**: 학습 동태 반영
- **설정**: `fim_update_freq=5`
- **효과**: 적응적 파라미터 선택

## 예상 결과

FisherTune 적용 시 다음과 같은 개선을 기대할 수 있습니다:

1. **NDCG@10**: 5-15% 향상
2. **HT@10**: 3-10% 향상
3. **학습 안정성**: 수렴 속도 개선
4. **일반화**: 도메인 간 전이 성능 향상

## 학습 효율성

- **추가 연산**: FIM 계산을 위한 gradient 저장 (약 10-20% 오버헤드)
- **메모리**: 대각 근사로 O(|θ|) 복잡도 유지
- **최적화**: Online Fisher estimation으로 배치당 추가 연산 최소화

## 실험 모니터링

학습 중 다음 정보가 로깅됩니다:

```
[FisherTune] Epoch 20: Threshold=0.7234, Active params=45, Ratio=35.16%
```

- **Threshold**: 현재 파라미터 선택 임계치
- **Active params**: 업데이트되는 파라미터 그룹 수
- **Ratio**: 전체 대비 활성 파라미터 비율

## 권장 설정

### 성능 최우선

```bash
python train_fishertune.py \
    --use_fishertune \
    --fishertune_warmup 15 \
    --shared_delta_min 0.6 \
    --specific_delta_min 0.08 \
    --use_adaptive_vi \
    --use_fisher_regularization \
    --perturbation_type combined
```

### 학습 속도 최우선

```bash
python train_fishertune.py \
    --use_fishertune \
    --fishertune_warmup 5 \
    --fim_update_freq 20 \
    --fim_method online
```

### 안정성 최우선

```bash
python train_fishertune.py \
    --use_fishertune \
    --fishertune_warmup 20 \
    --shared_delta_min 0.7 \
    --variational_tau 0.5 \
    --variational_gamma 0.2
```

## 문제 해결

### FIM이 모두 0인 경우
- 원인: Warmup 기간이 짧거나 학습이 수렴
- 해결: `--fishertune_warmup` 증가, `--fisher_damping` 조정

### 파라미터 선택이 너무 적은 경우
- 원인: Threshold가 너무 높음
- 해결: `--shared_delta_min`, `--specific_delta_min` 감소

### 학습 불안정
- 원인: 급격한 파라미터 변화
- 해결: `--use_adaptive_vi` 활성화, learning rate 감소

## 향후 개선 방향

1. **다중 GPU 지원**: 대규모 모델을 위한 분산 FIM 계산
2. **Layer-wise Scheduling**: 레이어별 차등 threshold 적용
3. **Meta-learning Integration**: 도메인 적응을 위한 메타 학습
4. **Attention-based Selection**: Attention 가중치 기반 파라미터 선택

## 참고 문헌

- FisherTune: Domain-Specific Fisher Information for Parameter Selection
- UniCDR: Universal Cross-Domain Recommendation
- Elastic Weight Consolidation for Continual Learning
- Natural Gradient Descent
