# neurone

## Requirements

Python 3.10

## Installation
```
pip install -e .
git lfs pull
```

## Normalization

Скрипт для получения `mean/std` к датасету, указанному в конфиге.

```bash
python tools/get_mean_std.py <path_to_config> --engine=<engine_name> --cuda=<cuda_number>
```

## Grad-cam

```bash
python tools/grad_cam.py <path_to_config> --weights-path=<path_to_model_weight> --dir=<dir_to_save_results> --engine=<engine_name> --cuda=<cuda_number>
```

Рабочий пример на a100:
```bash
python tools/grad_cam.py configs/inbreast.yaml --weights-path='/home/mammography_data/models/INBreast/EfficientNet_B3_full/experiment_2023-03-31 19:22:01.251700/model.pth' --dir='tmp/grad_cam' --engine=gpu --cuda='0'
```

Изображения будут находится в директории `<dir_to_save_results>`.

## Train
Run from the root of this project:
```bash
python tools/train_torch.py test/supplement/classification.json
```

## Base structure

```mermaid
graph LR
    BaseProcess[base] --> RunProcess[run]
    RunProcess --> InternalRun[_run]
    InternalRun --> RunLocal[_run_local]
    RunLocal --> OnKFoldStart[on_kfold_start]
    RunLocal --> RunKFold[run_kfold]
    RunLocal --> OnKFoldEnd[on_kfold_end]
    RunKFold --> OnExperimentStart[on_experiment_start]
    RunKFold --> RunExperiment[run_experiment]
    RunKFold --> OnExperimentEnd[on_experiment_end]
    RunExperiment --> OnEpochStart[on_epoch_start]
    RunExperiment --> RunEpoch[run_epoch]
    RunExperiment --> OnEpochEnd[on_epoch_end]
    RunEpoch --> OnDatasetStart[on_dataset_start]
    RunEpoch --> RunDataset[run_dataset]
    RunEpoch --> OnDatasetEnd[on_dataset_end]
    RunDataset --> OnBatchStart[on_batch_start]
    RunDataset --> RunBatch[run_batch]
    RunDataset --> OnBatchEnd[on_batch_end]
    RunBatch --> NotImplementedError[NotImplementedError]
    OnExperimentEnd --> SaveBestMetrics[save_best_metrics]
    OnKFoldEnd --> SummarizeKFold[summarize_kfold]
```

## Dataset hierarchy
```mermaid
graph LR
    Dataset(torch.utils.data.Dataset) --> BaseDataset(BaseDataset)
    BaseDataset --> BaseMammographyDataset(BaseMammographyDataset)
    BaseDataset --> BaseClassificationDataset(BaseClassificationDataset)
    BaseDataset --> BaseCOCODataset(BaseCOCODataset)
    BaseCOCODataset --> BaseSegmentationDataset(BaseSegmentationDataset)
    BaseCOCODataset --> BaseDetectionDataset(BaseDetectionDataset)
    BaseClassificationDataset --> QuadroMammographyDataset(QuadroMammographyDataset)
    BaseMammographyDataset --> QuadroMammographyDataset
    BaseClassificationDataset --> MultiLabelDataset(MultiLabelDataset)
    BaseDetectionDataset --> BinaryCOCOClassification(BinaryCOCOClassification)
    BaseClassificationDataset --> MammographyClassification(MammographyClassification)
    BaseMammographyDataset --> MammographyClassification
    BaseDetectionDataset --> COCODetection(COCODetection)
    PointsDataset(PointsDataset) --> HeatmapsDataset(HeatmapsDataset)
    Dataset --> HeatmapsDataset
    Dataset --> HeatmapModelDataset(HeatmapModelDataset)
    BaseDetectionDataset --> MammographyDetection(MammographyDetection)
    BaseMammographyDataset --> MammographyDetection
    HeatmapsDataset --> PrecomputedDataset(PrecomputedDataset)
    BaseSegmentationDataset --> SemanticSegmentationDataset(SemanticSegmentationDataset)
    BaseSegmentationDataset --> InstanceSegmentationDataset(InstanceSegmentationDataset)
    SemanticSegmentationDataset --> MammographySemanticSegmentation(MammographySemanticSegmentation)
    BaseMammographyDataset --> MammographySemanticSegmentation
    InstanceSegmentationDataset --> MammographyInstanceSegmentation(MammographyInstanceSegmentation)
    BaseMammographyDataset --> MammographyInstanceSegmentation
```

## TYFC
Run from the root of this project:
```
pytest
```
You are awesome!