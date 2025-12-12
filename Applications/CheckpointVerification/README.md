# Gradient Checkpointing Verification Application

## 개요

이 애플리케이션은 gradient checkpointing 구현의 정확성을 검증하기 위한 테스트 프로그램입니다. Forward pass에서 생성된 activation과 backward pass에서 recompute된 activation을 비교하여 구현이 올바른지 확인합니다.

## 빌드 방법

```bash
cd nntrainer
meson build
ninja -C build
```

실행 파일은 `build/Applications/CheckpointVerification/jni/nntrainer_checkpoint_verification`에 생성됩니다.

## 실행 방법

### 기본 실행

```bash
./build/Applications/CheckpointVerification/jni/nntrainer_checkpoint_verification
```

### 옵션 사용

```bash
./build/Applications/CheckpointVerification/jni/nntrainer_checkpoint_verification \
    --seq_len 64 \
    --model_dim 128 \
    --num_layers 4 \
    --epochs 3
```

### 사용 가능한 옵션

- `--seq_len <int>`: Sequence length (기본값: 32)
- `--model_dim <int>`: Model dimension (기본값: 64)
- `--num_heads <int>`: Number of attention heads (기본값: 4)
- `--num_layers <int>`: Number of transformer layers (기본값: 2)
- `--ffn_dim <int>`: FFN dimension (기본값: 256)
- `--batch_size <int>`: Batch size (기본값: 4)
- `--epochs <int>`: Number of epochs (기본값: 2)
- `--num_batches <int>`: Number of batches per epoch (기본값: 8)
- `--disable-verification`: 검증 기능 비활성화
- `--help, -h`: 도움말 표시

## 동작 방식

1. **모델 생성**: 간단한 transformer 모델 생성 (MHA + FFN)
2. **Checkpoint Block 설정**: 각 transformer layer를 checkpoint block으로 설정
3. **검증 활성화**: NetworkGraph의 검증 모드 활성화
4. **학습 실행**: 
   - Forward pass에서 checkpointed layer의 output 자동 저장
   - Backward pass에서 recompute 시 저장된 output과 비교
5. **통계 출력**: 검증 결과 통계 출력

## 예상 출력

### 성공적인 검증

```
========================================
Gradient Checkpointing Verification Test
========================================
[CONFIG] Test Configuration:
  - Batch size: 4
  - Sequence length: 32
  - Model dimension: 64
  - Number of heads: 4
  - Number of layers: 2
  - FFN dimension: 256
  - Epochs: 2
  - Number of batches: 8
  - Verification: ENABLED

[TEST] Creating test model with gradient checkpointing...
[TEST] Adding checkpoint blocks...
[TEST] Added checkpoint block for layer0 (10 layers)
[TEST] Added checkpoint block for layer1 (10 layers)

[TEST] Configuring model...
[TEST] Compiling model...
[TEST] Initializing model...

[TEST] Enabling checkpoint verification...
[INFO] Gradient checkpointing verification enabled
[TEST] Checkpoint verification ENABLED

[TEST] Setting up dataset...

[TEST] Starting training...
========================================
[DEBUG] Saved forward outputs for layer 'layer0/ln1' (1 tensors)
[DEBUG] Saved forward outputs for layer 'layer0/mha' (1 tensors)
...
[DEBUG] Recomputing layer 'layer0/ln1'
[DEBUG] Output match for layer 'layer0/ln1' tensor[0]: max_diff=0.0000000000, avg_diff=0.0000000000
[INFO] ✓ Verification PASSED for layer 'layer0/ln1'
...
========================================
[TEST] Training completed successfully!

[TEST] Printing verification statistics...
========== Gradient Checkpointing Verification Statistics ==========
Total verifications: 32
Passed: 32
Failed: 0
Pass rate: 100.00%
Max difference: 0.0000000123
Avg difference: 0.0000000045
====================================================================

========================================
Test completed successfully!
========================================
```

### 검증 실패 시

```
[ERROR] Output mismatch for layer 'layer1/fc1' tensor[0]: max_diff=0.1234567890, avg_diff=0.0567890123
[ERROR] ✗ Verification FAILED for layer 'layer1/fc1'
...
========== Gradient Checkpointing Verification Statistics ==========
Total verifications: 32
Passed: 31
Failed: 1
Pass rate: 96.88%
Max difference: 0.1234567890
Avg difference: 0.0012345678
====================================================================
```

## 검증 기준

- **절대 tolerance**: `1e-6`
- **상대 tolerance**: `1e-5`
- **판정**: `|saved - recomputed| <= abs_tol + rel_tol * |saved|`

모든 element가 이 기준을 만족하면 PASS, 하나라도 실패하면 FAIL

## 문제 해결

### 검증 실패 시 체크리스트

1. **Random seed**: Dropout 등 random 연산 사용 시 seed 고정 필요
2. **In-place 연산**: Checkpointed layer에서 in-place 연산 비활성화 확인
3. **Batch Normalization**: Running statistics 업데이트로 인한 차이 가능
4. **Numerical precision**: FP16 사용 시 tolerance 조정 필요

### 메모리 부족

작은 설정으로 테스트:
```bash
./nntrainer_checkpoint_verification --num_layers 1 --seq_len 16 --batch_size 2
```

## 코드 구조

```
CheckpointVerification/
├── jni/
│   ├── main.cpp          # 메인 테스트 프로그램
│   └── meson.build       # 빌드 설정
└── README.md             # 이 파일
```

## 참고

- 이 애플리케이션은 디버깅/검증 목적으로만 사용하세요
- 프로덕션 환경에서는 검증 기능을 비활성화하세요 (`--disable-verification`)
- 검증 기능은 메모리와 성능에 영향을 줍니다
