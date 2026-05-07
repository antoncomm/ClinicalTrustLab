# ClinicalTrustLab (CTL)

![`CTL schema`](images/scheme.png)

This repository provides a PyTorch-based framework for running mammography (and other medical imaging) experiments:
- Training / evaluation loops
- Dataset wrappers and utilities
- Configurable transforms
- Hydra-based config composition

## Table of contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configs](#configs)
  - [Hydra basics in this repo](#hydra-basics-in-this-repo)
  - [How to create your own config](#how-to-create-your-own-config)
- [Add your dataset](#add-your-dataset)
- [Add your model](#add-your-model)
- [Tutorials and examples](#tutorials-and-examples)



## Installation

```bash
git clone https://github.com/antoncomm/ClinicalTrustLab
cd ClinicalTrustLab

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```



## Quick start

### 1) Prepare your data
More details in [Add your dataset](#add-your-dataset).

For a simple **classification** pipeline we expect a CSV annotation with at least two columns:

- `image` — path to an image **relative** to `image_folder`
- `label` — class name (after applying `class_map` it must become a number)

Example:

```csv
image,label
patient_001/image1.dicom,BENIGN
patient_001/image2.dicom,MALIGNANT
```

### 2) Choose or create a config
More details in [Configs](#configs).

A good starting point is the composed Hydra config:
- [`configs/efficientnet_cls_augs.yaml`](./configs/efficientnet_cls_augs.yaml)

### 3) Run training

From the repository root:

```bash
python tools/train_torch.py configs/examples/inbreast_efficientnet.yaml \
  --engine gpu \
  --cuda 0 \
  --name test_run \
  --workers 8 \
  --seed 42
```

Outputs are written to `./experiments/test_run/`:

- `checkpoints/` — per-epoch and best model weights
- `logs/` — tracker logs (e.g. TensorBoard)
- `graphs/` — metric curves / plots



## Configs

A “full” experiment config is a regular YAML dictionary. Key sections:

- `mode`: `train` or `test`
- `model`: what to create and how to load weights
- `datasets`: one or multiple datasets (combined via `BaseConcatDataset`)
- `transforms`: Albumentations pipeline
- `run`: batch size, epochs, optimizer, metrics, tracker, k-fold, etc.

### Hydra basics in this repo

The folder [`configs/`](./configs/) contains **Hydra** configuration groups that can be composed via `defaults:`:
- [`configs/model/`](./configs/model/)
- [`configs/datasets/`](./configs/datasets/)
- [`configs/run/`](./configs/run/)
- [`configs/transforms/`](./configs/transforms/)

Hydra entrypoint in this repo:
- [`configs/config.yaml`](./configs/config.yaml) — the base Hydra config

#### Hydra overrides (how to change params from CLI)

When running Hydra-based scripts, you can override any nested key:

```bash
python tools/get_mean_std.py run.engine_name=gpu run.cuda_visible_devices=0 run.batch_size=64
```

To override lists / dicts, follow Hydra/OmegaConf syntax:

```bash
python tools/get_mean_std.py transforms.augmentations.Normalize.mean=[0.12,0.12,0.12]
```

### How to create your own config

1) Create a new top-level experiment YAML in `configs/`, e.g. `configs/my_experiment.yaml`:

```yaml
# configs/my_experiment.yaml
mode: train

defaults:
  - model: efficientnet
  - datasets:
      - inbreast_cls
  - transforms: simple_augs
  - run: classification
  - _self_

# You can override any composed values below:
run:
  num_epochs: 10
  batch_size: 32
```

2) Run a Hydra tool with this experiment config by overriding `config_name` in the decorator (or create a dedicated script).
In the current repo, tools are hardwired to `config_name="config"` (see `@hydra.main(...)`),
so the simplest pattern is: **add your experiment to `configs/config.yaml`** as a default.

For example, in `configs/config.yaml` replace the default experiment:

```yaml
defaults:
  - _self_
  - my_experiment
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled
```

Now tools will run with your experiment by default:

```bash
python tools/get_mean_std.py
python tools/grad_cam.py
```



## Add your dataset

### Default dataset layout (recommended)

By default, the repository already contains convenient dataset classes that assume the following structure:

```text
your_dataset/
  data/
    patient_1/
      image1.dicom
      image2.dicom
      image3.dicom
      image4.dicom
    patient_2/
      image1.dicom
      image2.dicom
      image3.dicom
      image4.dicom
  train.csv
  valid.csv
  test.csv
```

This layout is recommended because existing dataset classes can:
- read CSV annotations (`train.csv`, `valid.csv`, `test.csv`)
- build full image paths as `dataset_dir / data / <relative_path_from_csv>`
- apply shared preprocessing and collate logic

### Can I use a different dataset format?

Yes — you can use **any** folder structure / annotation format, but then you will need to:
- implement your own `torch.utils.data.Dataset` (or subclass our base classes)
- implement the parsing/processing logic (paths, labels, metadata, etc.)
- (optionally) provide a custom `collate_fn`

### 1) Implement a dataset class

1. Create a file, e.g.: `src/ctl/data/datasets/my_dataset.py`
2. Inherit from a base class:
   - `BaseDataset` — full manual control
   - `BaseClassificationDataset` — CSV-based classification
   - `BaseMammographyDataset` — mammography-specific logic (e.g. windowing / mask crop)
3. Implement at least:
   - `__len__`
   - `__getitem__`
   - `collate_fn`
   - (if you do not use the provided `_setup_annotation`) `_setup_annotation`

**Important:** `setup_data()` resolves dataset classes by name using `globals()` after
`from ctl.data.datasets import *`.  
Therefore your dataset class **must** be imported in `src/ctl/data/datasets/__init__.py`.

### 2) Export the dataset in `__init__.py`

Open `src/ctl/data/datasets/__init__.py` and add your import and class name:

```python
__all__ = [
    "MammographyClassification",
    "BaseConcatDataset",
    "BaseClassificationDataset",
    "MyDataset",
]

from .my_dataset import MyDataset
```

### 3) Add a dataset config

Create a file, e.g. `configs/datasets/my_dataset.yaml`:

```yaml
my_dataset:
  dataset: MyDataset
  dataset_dir: /path/to/your_dataset
  ann_files:
    train: ${datasets.my_dataset.dataset_dir}/train.csv
    valid: ${datasets.my_dataset.dataset_dir}/valid.csv
    test:  ${datasets.my_dataset.dataset_dir}/test.csv
  kwargs: {}
  train_kwargs: {}
  valid_kwargs: {}
  test_kwargs: {}
  ignore_classes: []
  class_map: {}
```

Then include it in a Hydra experiment:

```yaml
defaults:
  - datasets:
      - my_dataset
```



## Add your model

### 1) Implement the model

1. Create a module, e.g.: `src/ctl/models/classification/my_model.py`
2. Implement a `torch.nn.Module`.

### 2) Export the model in `ctl.models`

`initialize_model()` calls `getattr(ctl.models, config_model["model_name"])`, so you must:

1) Add an import to `src/ctl/models/__init__.py`  
2) Add the name to `__all__`

Example:

```python
__all__ = [
    "EfficientNet",
    "MyModel",
]

from .classification.efficientnet import EfficientNet
from .classification.my_model import MyModel
```

### 3) Add a model config

Create `configs/model/my_model.yaml`:

```yaml
runner: SimpleClassificationRunner
model_name: MyModel
model_kwargs:
  num_classes: 1
weights_path: null
thresholds:
  cls: 0.5
```

Now you can refer to the model via `model: my_model` in a Hydra experiment config.



## Run attacks

You can enable training data corruption or noise additions by specifying the `attack` config group in your experiment.

```yaml
# configs/my_experiment.yaml
mode: train

defaults:
  - model: efficientnet
  - datasets:
      - inbreast_cls
  - transforms: simple_augs
  - attack: pgd # enable attack here
  - run: classification
  - _self_
```

Available attacks are defined in `configs/attack/`.
Each attack has its own YAML file inside `configs/attack/`.

For example, `configs/attack/pgd.yaml` may contain:

```yaml
type: PGDAttack
attack_runner: AttackMultiClassificationRunner
attack_params:
  nb_classes: 2
  max_iter: 5
  eps: 0.05
  eps_step: 0.01
  verbose: False
  is_color: True
```

You can modify these values directly in the YAML file.

For evasion attacks, you must use a compatible runner:
   - `SimpleClassificationRunner` — for standard single-label classification
   - `AttackMultiClassificationRunner` — for multi-label or specific attack-aware setups

This is defined in your model config in `type` field.
