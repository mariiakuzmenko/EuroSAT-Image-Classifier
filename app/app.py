import PIL.Image
import numpy as np
import pandas as pd
from shiny.express import input, render, ui
from shiny import reactive
import torch
from torch import nn
from torchvision import transforms

_ = torch.manual_seed(42) # assigned return value, so shiny would not try to render it

# Defining SatelliteCNN architecture which inherits nn.Module
class SatelliteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 Convolutional layers (feature extractors)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # In: 3 channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Classifier head (Maps 4096 flattened features down to the 10 target classes)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

        self.flatten = nn.Flatten() # Flatten the output of the convolutional layers
        self.relu = nn.ReLU() # Introduce non-linearity
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # Downsample by 2
        self.dropout = nn.Dropout(0.5) # Dropout (50% probability, for regularization, only during training)

        # Batch normalization stabilizes the learning step across color channels
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm2d(128)
        self.batch4 = nn.BatchNorm2d(256)

    def forward(self, x):
        # 1st stage
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 2nd stage
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 3rd stage
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4th stage
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 5th stage
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

test_transf = transforms.Compose([
    transforms.ToTensor()
])

model = SatelliteCNN().to('cpu')   # create an instance of the model

# Loading the weights and Evaluation mode
state_dict = torch.load('best_model_params.pth', map_location='cpu', weights_only=True)
_ = model.load_state_dict(state_dict)
_ = model.eval()

# Helper reversed label dictionary
reverse_label_dict = {
        0 : 'HerbaceousVegetation',
        1 : 'AnnualCrop',
        2 : 'Residential',
        3 : 'Pasture',
        4 : 'Industrial',
        5 : 'River',
        6 : 'Highway',
        7 : 'Forest',
        8 : 'PermanentCrop',
        9 : 'SeaLake'
    }
@reactive.calc  # Caches the resulting probability distribution for other components.
@reactive.event(input.predict)  # Only calculates when the button 'Predict' is clicked
def run_model():

    file_info = input.image()
    if file_info is None:
        return None

    # Load and preprocess image tensor
    raw_image = PIL.Image.open(file_info[0]["datapath"]).convert("RGB")
    image_tensor = test_transf(raw_image).unsqueeze(0).to('cpu')  # Create a batch of one

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities[0].cpu().numpy()

# Main container for page
with ui.card(full_screen=True, height="800px"):
    ui.card_header("Satellite Image Classifier")   # Card title

    with ui.layout_sidebar():  # Provides sidebar layout

        with ui.sidebar():   # Sidebar container
            ui.input_file(id="image", label="Upload image")   # Input file component

            # Button for prediction, with green color
            ui.input_action_button(id="predict", label="Predict class", class_="btn-success")

        with ui.layout_column_wrap(width=1/2):   # Provides equal space for other cards in column manner
                                        # Main container for predictions etc.

            with ui.card():  # Container for image
                ui.card_header("Uploaded Image")

                @render.image(delete_file=False)  # Render image function
                def render_image():

                    file_info = input.image()

                    if file_info is None:
                        return None

                    # Extract the path of the uploaded image
                    img_path = file_info[0]["datapath"]

                    # Hand @render.image the dictionary format it demands
                    return {
                        "src": img_path,
                        "width": "auto",  # Forces photos to scale to the card width
                        "height": "400"
                    }

            with ui.card():  # Container for table and prediction
                ui.card_header("Results")

                with ui.card():  # Container for table of probabilities
                    ui.card_header("Class Probabilities")

                    @render.data_frame   # Because the table is dataframe
                    def predict_df():
                        probs = run_model()

                        if probs is None:
                            return None

                        df = pd.DataFrame({    # Dataframe with Classes and Probabilities columns
                            "Class": [reverse_label_dict[i] for i in range(10)], # List of classes
                            "Probability": np.round(probs.astype(float), 4)
                        })                 # Take probabilities, to cpu, to numpy array,
                                           # type float (for correct rounding), with 4 decimal places

                        df = df.sort_values(by='Probability', ascending=False)  # Sort df by Probabilities column,
                                                                                # descending
                        return df

                    ui.div(class_="mt-3")   # Add spacing between table and prediction

                with ui.card():  # Container for prediction
                    ui.card_header("Prediction")

                    @render.text   # Render prediction text
                    def predicted_class():
                        probs = run_model()

                        if probs is None:
                            return None

                        predicted = np.argmax(probs)  # Because converted to numpy prior
                        confidence = probs[predicted] * 100

                        return f"Predicted Class: {reverse_label_dict[predicted]} (Confidence: {confidence:.2f}%)"
