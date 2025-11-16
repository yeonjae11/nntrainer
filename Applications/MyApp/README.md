MyApp (Random Data Baseline)
============================

Overview
- A minimal example that trains on random data to quickly compare optimizers.
- Assumes FP32-only build/runtime.

Build
1) From repository root:
```
meson setup build -Dbuildtype=release -Denable-app=true
meson compile -C build -j"$(nproc)"
```

Run Examples
- Same configuration; switch only the optimizer:
```
cd build/Applications/MyApp/jni
./nntrainer_myapp --opt=lion  --wd=0     --bs=16 --db=32 --epochs=5 --lr=0.001
./nntrainer_myapp --opt=lion  --wd=0.01  --bs=16 --db=32 --epochs=5 --lr=0.001
./nntrainer_myapp --opt=adam             --bs=16 --db=32 --epochs=5 --lr=0.001
./nntrainer_myapp --opt=adamw            --bs=16 --db=32 --epochs=5 --lr=0.001
./nntrainer_myapp --opt=sgd              --bs=16 --db=32 --epochs=5 --lr=0.001
./nntrainer_myapp --opt=sgd              --bs=16 --db=32 --epochs=5 --lr=0.0005
```

Options
- `--opt=lion|adam|adamw|sgd` : optimizer (default: lion)
- `--wd=<float>`              : weight decay (used by Lion)
- `--epochs=<int>`            : number of epochs
- `--db=<int>`                : number of batches (iterations)
- `--bs=<int>`                : batch size
- `--lr=<float>`              : learning rate

Output
- Prints L2 norm of weights before/after training, delta L2, and per-epoch training loss.

Notes
- Because data is random, loss curves are better suited to compare update magnitude/consistency rather than convergence. For quantitative comparison, use `MyApp_MNIST`.
