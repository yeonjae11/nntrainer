MyApp_MNIST (INI + MNIST Dataset)
=================================

Overview
- Uses the MNIST example (INI-configured) to compare optimizers under identical settings.
- Meson build copies `mnist.ini` and `mnist_trainingSet.dat` next to the executable.
- Assumes FP32-only build/runtime.

Build
1) From repository root:
```
meson setup build -Dbuildtype=release -Denable-app=true
meson compile -C build -j"$(nproc)"
```

Resource Paths
- Prefers local `./mnist.ini` and `./mnist_trainingSet.dat` in the working directory (copied at build time).
- If not found, it auto-searches up ancestor directories for `Applications/MNIST/res/*`.
- If still not found, specify paths explicitly with `--config` and `--data`.

Run Examples
- Compare optimizers with the same settings:
```
cd build/Applications/MyApp_MNIST/jni
./nntrainer_myapp_mnist --opt=lion  --lr=0.001 --wd=0.01 --epochs=100 --bs=32
./nntrainer_myapp_mnist --opt=adam  --lr=0.001              --epochs=100 --bs=32
./nntrainer_myapp_mnist --opt=adamw --lr=0.001 --wd=0.01     --epochs=100 --bs=32
./nntrainer_myapp_mnist --opt=sgd   --lr=0.001              --epochs=100 --bs=32
```
- Run from an arbitrary directory with explicit paths:
```
./build/Applications/MyApp_MNIST/jni/nntrainer_myapp_mnist \
  --config=Applications/MNIST/res/mnist.ini \
  --data=Applications/MNIST/res/mnist_trainingSet.dat \
  --opt=lion --lr=0.001 --wd=0.01 --epochs=5 --bs=32
```

Options
- `--config=<path>`           : INI path (auto-discovered if not provided)
-,`--data=<path>`             : dataset path (auto-discovered if not provided)
- `--opt=lion|adam|adamw|sgd` : optimizer (default: lion)
- `--lr=<float>`              : learning rate (default: 1e-3)
- `--wd=<float>`              : weight decay (used by Lion)
- `--epochs=<uint>`           : number of epochs (default: 5)
- `--bs=<uint>`               : batch size (default: 32)
- `--train_size=<uint>`       : number of training samples (default: 100)
- `--val_size=<uint>`         : number of validation samples (default: 100)

Output
- Prints L2 norm of weights before/after training, delta L2, and standard NNTrainer training logs (epoch losses).

Notes
- MNIST exposes optimizer differences in loss curves more clearly than random data. Recommended for validating the new Lion implementation.
