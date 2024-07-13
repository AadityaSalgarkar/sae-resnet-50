import torch
import torchvision.models as models
from collections import OrderedDict


def get_resnet50_with_hooks():
    # Load pre-trained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()  # Set the model to evaluation mode

    # Dictionary to store activations
    activations = OrderedDict()

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    # Register forward hook for layer3
    model.layer3.register_forward_hook(get_activation("layer3"))

    return model, activations


def get_layer3_activations(model, activations, input_tensor):
    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    # Return the activations of layer3
    return activations["layer3"]


def get_random_layer3_activation(random_input=None):
    if random_input is None:
        # Create a random input tensor
        random_input = torch.randn(
            1, 3, 224, 224
        )  # Assuming standard ImageNet input size

    # Get the model and activations dictionary
    model, activations = get_resnet50_with_hooks()

    # Get layer3 activations for the random input
    layer3_activation = get_layer3_activations(model, activations, random_input)

    return layer3_activation


import matplotlib.pyplot as plt
import seaborn as sns


def visualize_activation(activation, channel):
    # Extract the specified channel
    channel_activation = activation[0, channel, :, :].cpu().numpy()

    # Create a new figure
    plt.figure(figsize=(10, 8))

    # Create a heatmap using seaborn
    sns.heatmap(channel_activation, cmap="viridis", cbar=True)

    plt.title(f"Activation Heatmap for Channel {channel}")
    plt.xlabel("Width")
    plt.ylabel("Height")

    # Show the plot
    plt.show()


# Visualize the activation for channel 15
# acts = get_random_layer3_activation()
# print(acts.shape)
# visualize_activation(acts, 15)

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def process_image_and_show_activations(input_image):
    # Resize and preprocess the input image
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image).unsqueeze(0)

    # Get the model and activations
    model, activations = get_resnet50_with_hooks()

    # Get layer3 activations for the input image
    layer3_activation = get_layer3_activations(model, activations, input_tensor)

    # Generate heatmaps for all channels
    num_channels = layer3_activation.shape[1]
    heatmaps = []
    for channel in range(num_channels):
        print(f"channel {channel} done")
        channel_activation = layer3_activation[0, channel, :, :].cpu().numpy()
        plt.figure(figsize=(5, 5))
        sns.heatmap(channel_activation, cmap="viridis", cbar=False)
        plt.title(f"Channel {channel}")
        plt.axis("off")

        # Convert plot to image
        fig = plt.gcf()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()

        heatmaps.append(img)

    return heatmaps


# Create Gradio interface
iface = gr.Interface(
    fn=process_image_and_show_activations,
    inputs=gr.Image(type="pil"),
    outputs=gr.Gallery(label="Layer 3 Activation Heatmaps"),
    title="ResNet50 Layer 3 Activation Visualizer",
    description="Upload an image to see the activation heatmaps for all channels in layer 3 of ResNet50.",
)

# Launch the app
iface.launch()
