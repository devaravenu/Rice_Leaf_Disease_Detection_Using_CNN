# Rice Leaf Disease Detection Using CNN

An educational computer vision project that explores rice leaf images and compares two convolutional neural network (CNN) classifiers: a baseline model and a model trained with data augmentation.

## Project objective

The notebook builds an end-to-end image classification workflow. It inspects the image dataset, preprocesses images, trains and compares CNNs, evaluates the selected model, and demonstrates a prediction on a dataset image.

The project is a learning prototype. Its small dataset and test set are not sufficient to establish reliable performance in real field conditions.

## Disease classes and dataset

The notebook expects a `Data` directory beside the notebook's working directory, with one folder per class:

```text
Data/
├── Bacterial leaf blight/
├── Brown spot/
└── Leaf smut/
```

The notebook's recorded audit found 119 readable images: 40 Bacterial leaf blight, 40 Brown spot, and 39 Leaf smut. It re-counts files when run, so the current dataset contents are the source of truth.

**Label note:** the notebook reports that an assignment description mentions “Leaf blast,” while the supplied dataset folder is “Leaf smut.” These are different labels; confirm the intended class and do not relabel the images without verifying them.

The dataset is not included here. Add it only if you have permission to redistribute it. Otherwise, provide the dataset source and instructions for obtaining it, and keep the image files out of the Git repository.

## Workflow

The notebook includes:

1. Image inventory, class counts, sample visualization, dimensions, and file readability checks.
2. SHA-256 checks for exact duplicate files.
3. RGB conversion, resizing to 128 × 128 pixels, and normalization to the 0–1 range.
4. A stratified train, validation, and test split (approximately 70%, 15%, and 15%).
5. A baseline CNN and a comparable CNN with training-time horizontal flip, small rotation, and zoom augmentation.
6. Early stopping and training-curve visualization.
7. Validation-based model selection, followed by held-out test evaluation using accuracy, macro precision/recall/F1, ROC-AUC, and confusion matrices.
8. A sample prediction demonstration. Its confidence score is the model output, not a measure of accuracy.

Augmentation is applied dynamically during training; it does not create additional image files.

## Recorded results

The notebook reports these results for its recorded test split:

| Metric | Without augmentation | With augmentation |
|---|---:|---:|
| Accuracy | 61.11% | 72.22% |
| Macro precision | 59.52% | 72.70% |
| Macro recall | 61.11% | 72.22% |
| Macro F1-score | 59.49% | 72.11% |
| Macro ROC-AUC | 75.46% | 89.35% |

The test set contains only 18 images (six per class), so a small number of changed predictions can substantially move these metrics. Results can also vary across splits and training runs. Treat these figures as the notebook's recorded experiment, not a performance guarantee.

## Requirements

- Python 3.10 or newer is recommended.
- Jupyter Notebook or JupyterLab.
- Python packages: TensorFlow, NumPy, pandas, Matplotlib, Seaborn, Pillow, and scikit-learn.

Install the packages from the project folder:

```bash
python -m pip install jupyter numpy pandas matplotlib seaborn pillow tensorflow scikit-learn
```

If you add a `requirements.txt`, install with `python -m pip install -r requirements.txt` instead.

## Run the notebook

1. Put the notebook in a `notebooks/` folder (recommended) and arrange the dataset as described above. Run Jupyter with the project root as its working directory so the relative `Data` path resolves correctly; otherwise, update the notebook's `data_path`/`DATA_DIR` to the dataset location.
2. Start Jupyter:

   ```bash
   jupyter lab
   ```

3. Open `notebooks/Rice_Leaf_Disease_Detection_Using_CNN.ipynb` and run the cells from top to bottom.
4. Check the notebook output for the observed image counts, training histories, model comparison, and saved-model path.

CNN training can take longer without a supported GPU. Exact metrics may differ with TensorFlow versions, hardware, and random initialization.

## Suggested repository layout

```text
Rice-Leaf-Disease-Detection/
├── README.md
├── requirements.txt
├── notebooks/
│   └── Rice_Leaf_Disease_Detection_Using_CNN.ipynb
└── Data/                         # local dataset; add only if redistribution is allowed
    ├── Bacterial leaf blight/
    ├── Brown spot/
    └── Leaf smut/
```

## Limitations and responsible use

- The dataset is very small, and the held-out test set has only 18 images.
- A random image prediction may come from the training, validation, or test split; it is a demonstration, not an independent test.
- Softmax scores are not calibrated probabilities or expert diagnoses.
- Image-level splits may not prevent leakage between images from the same plant, session, or source.
- Field use would require more correctly labelled images and evaluation on a larger, independent dataset collected across conditions, rice varieties, devices, and disease stages.
- The reported results are preliminary and should not be used to make agricultural treatment decisions.
## Conclusion

This project demonstrates an end-to-end CNN workflow for classifying rice leaf images into Bacterial leaf blight, Brown spot, and Leaf smut. It compares a baseline model with one trained using data augmentation and records higher test metrics for the augmented model on the notebook's current split. With only 119 readable images and 18 test images, those results are preliminary and may change substantially with more data or a different split. The model is best treated as an educational prototype; larger, independently collected and verified datasets are needed before considering real-world use.
