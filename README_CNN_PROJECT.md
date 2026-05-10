# Stage 3 PyTorch CNN Project

This project trains CNN image classifiers for the three pickle datasets provided in the assignment:

- `MNIST`
- `ORL`
- `CIFAR`

The original dataset files are not modified. All new outputs are written into the `outputs/` folder.

## Files

- `script_data_loader.py`: original course loader provided with the data
- `cnn_project/data.py`: reusable PyTorch dataset class for the pickle files
- `cnn_project/model.py`: configurable CNN model and experiment settings
- `cnn_project/train.py`: training, evaluation, plotting, and result-table utilities
- `run_experiments.py`: main script that trains models and saves outputs

## What the code does

1. Loads each pickle dataset.
2. Detects the image shape automatically.
3. Detects the number of classes automatically.
4. Converts images into normalized PyTorch tensors.
5. Trains CNN models for each dataset.
6. Saves learning curves for loss and accuracy.
7. Saves evaluation metrics in a table.
8. Runs multiple CNN configurations for comparison.

## Experiment configurations

The script includes four experiments:

- `baseline`: two convolution layers
- `deeper`: three convolution layers
- `larger_kernel`: uses `5x5` kernels
- `wider_hidden`: uses a larger fully connected hidden layer

## Run everything

```bash
python run_experiments.py
```

This trains:

- one model family on `MNIST`
- one model family on `ORL`
- one model family on `CIFAR`

with all four experiment settings.

## Helpful options

Run only one dataset:

```bash
python run_experiments.py --datasets ORL
```

Run only one configuration:

```bash
python run_experiments.py --experiments baseline
```

Quick smoke test with fewer samples:

```bash
python run_experiments.py --datasets MNIST ORL CIFAR --experiments baseline --epochs 1 --max-train-samples 128 --max-test-samples 64
```

## Output files

After training, the script saves:

- `outputs/models/*.pt`: trained model weights
- `outputs/history/*.json`: training history and dataset metadata
- `outputs/figures/*_learning_curves.png`: loss/accuracy plots
- `outputs/results/evaluation_results.csv`: evaluation table
- `outputs/results/evaluation_results.md`: Markdown version of the table

## Notes for ORL

The ORL images are stored as `112x92x3`, but the three RGB channels contain the same grayscale values. The dataset class detects this automatically and uses one channel for the CNN input.
