# Playing Card Classification with Deep Learning

**Author:** Beñat Ilzarbe Daza

This project explores deep learning approaches to classify playing card images into **53 classes: 52 standard cards and one joker class**.

The notebook documents the complete experimental process, including baseline models, dataset expansion, transfer learning, hyperparameter optimization, and model ensembles.

## Dataset

The final training setup uses the original dataset supplemented with 15 manually created joker images:

| Split | Images |
|---|---:|
| Training | 7,636 |
| Validation | 268 |
| Test | 265 |

The additional jokers were distributed between training (12) and validation (3). **The original test set remained unchanged**, with five images per class.

An earlier experiment also incorporated images from the [Card Arithmetic Dataset — Imperial College London](https://helix.imperial.ac.uk/records/2q13m-kb120). This expansion did not improve performance on the original test set and was excluded from the final training setup.

## Experimental Process

1. Establish baselines using a custom CNN and MobileNetV2.
2. Expand the dataset and compare pretrained architectures.
3. Return to the original data while retaining the additional jokers.
4. Compare MobileNetV2, ResNet50V2, and EfficientNetB0.
5. Refine training using data augmentation, dropout, and selective fine-tuning.
6. Train with images resized to 320 × 320 pixels.
7. Run 20 Optuna trials: 15 completed and five pruned.
8. Evaluate model ensembles and test-time augmentation (TTA).
9. Compare additional combinations and retain the best observed result.

## Results

| Approach | Original Test Accuracy |
|---|---:|
| Baseline MobileNetV2 | 86.42% |
| MobileNetV2 with additional jokers | 87.92% |
| Further fine-tuned MobileNetV2 | 90.19% |
| EfficientNetB0 with revised training at 320 × 320 | 95.85% |
| Three-model Optuna ensemble with TTA | 98.49% |
| Final weighted ensemble | **99.25%** |

The final ensemble correctly classified **263 of 265 test images**, including all five jokers. This represents **34 additional correct predictions** and a **12.83 percentage point improvement** over the baseline.

## Final Ensemble

The final solution averages the predicted class probabilities from four EfficientNetB0 models:

| Optuna Trial | Weight | Inference Views |
|---|---:|---|
| 12 | 46.3125% | Original and 180° rotation |
| 13 | 4.875% | Original and 180° rotation |
| 16 | 2.5000% | Original only |
| 17 | 46.3125% | Original and 180° rotation |

For models using TTA, predictions from the two views are averaged before applying the ensemble weights. The ensemble requires seven predictions per image and does not involve additional training.

Additional TTA using small rotations did not improve accuracy and was discarded.

## Evaluation Note

The original test images were never added to training or validation. However, the final ensemble weights and inference options were selected using performance on this test set.

Therefore, **99.25% is the best observed accuracy optimized on this specific test set**, rather than an independent estimate of generalization. A separate, untouched evaluation set would be needed to assess the final solution independently.

## Technologies

- Python
- TensorFlow / Keras
- Optuna
- NumPy and pandas
- scikit-learn
- Matplotlib and Seaborn
- OpenCV and Pillow
- Google Colab and Google Drive

## Running the Notebook

Open `Beñat_Ilzarbe_Prediccion_Cartas.ipynb` in Google Colab and select a GPU runtime.

Before running:

1. Mount Google Drive.
2. Update the project directory to match your Drive structure.
3. Provide the datasets and additional joker images referenced by the notebook.
4. Execute the cells in order.

Training and hyperparameter optimization can take several hours. Saved outputs are included so the experimental results can be reviewed without rerunning the full notebook. Reusing the existing Optuna study also requires its saved study files and model checkpoints.

## Main Conclusions

Adding more data did not automatically improve performance. The strongest improvements came from refining preprocessing, augmentation, fine-tuning, and hyperparameters, followed by combining models with complementary predictions.

EfficientNetB0 produced the strongest results among the configurations tested. The final ensemble improved accuracy further while requiring no additional training.
