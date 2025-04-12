# FYP Federated Learning

## Code Structure

- **demo**: Run model predictions on sample videos or images
- **models**: EfficientNet and XceptionNet based models for local, centralised, and federated learning setups
- **preprocessing**: Preprocesses using frame sampling, face detection and normalisation
- **testing**: Scripts for model evaluation on test sets
- **training**: Scripts for model training and validation

## How to Run Inference

1. Navigate to the `demo/` folder.

2. Open either:

   - `test_on_image.ipynb` – for testing on a single image
   - `test_on_video.ipynb` – for testing on a video clip

3. Modify the following parameters in the notebook:

   - **model_path**: path to trained model checkpoint (`.pth`)
   - **image_path/video_path**: path to input media file
   - **Model architecture**: choose between EfficientNet or XceptionNet

4. Run all cells in the notebook to see the model's predictions.
