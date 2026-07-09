# EuroSAT AI Image Classifier 

A Convolutional Neural Network (CNN) built with PyTorch that classifies satellite imagery into 10 distinct topological terrains. 

## The Architecture
* **Framework:** PyTorch
* **Model:** 4-Block Custom CNN (Feature Extractor + Linear Classifier)
* **Techniques Used:** Batch Normalization, Dropout Regularization (0.5), Early Stopping, StepLR Learning Rate Decay.

## Results
The model was trained on the EuroSAT dataset and achieved high accuracy on the validation set. 

`![Training Curves](assets/plots/training_curves.png)`

`![Confusion Matrix](assets/plots/confusion_matrix.png)`

## How to Run Locally
*Note: The specific dataset split used to train this model was provided securely by Johannes Kepler University for an academic examination and is not hosted in this repository.*

1. Clone the repository.
2. Provide your own dataset of categorized satellite images. 
3. Place the images in a root `data/` folder, with sub-folders named by class (e.g., `data/Forest/`, `data/Highway/`).
4. Install dependencies: `pip install -r requirements.txt`
5. Run the Jupyter Notebook.
