import os
import torch
import numpy as np
import cv2  
from torchvision import transforms,models
from PIL import Image
from django.conf import settings


# Load your pre-trained model (replace with your model's class if necessary)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Path to the saved model file
# model_path = os.path.join(os.path.dirname(__file__), "model.pth")

# # Load the model (ensure the architecture matches your saved model)
# model = torch.load("C:\Bone_cancer\Bone\ML\Bone_cancer.pth", map_location=device)
# model.eval()  # Set the model to evaluation mode
# model.to(device)


# Define your model architecture (e.g., ResNet101 in this case)
model = models.resnet101(pretrained=False)  # Use the same architecture as the one used for training
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Update the final layer to match your 2 classes ('cancer', 'healthy')

# Load the state dictionary into the model
state_dict = torch.load(r"ML\bone\Bone_cancer.pth", map_location=device)
# state_dict = torch.load("C:\Bone_cancer\Bone\ML\Bone_cancer.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode
model.to(device)

output_folder_path=os.path.join(settings.MEDIA_ROOT,"output")
input_folder_path=os.path.join(settings.MEDIA_ROOT,"bone")

def process_image(image_path=input_folder_path, output_folder=output_folder_path):
    """
    Processes an image and saves the Grad-CAM visualization to a folder.

    Args:
        image_path (str): Path to the input image.
        output_folder (str): Folder to save Grad-CAM images.

    Returns:
        str: Predicted class.
        str: Path to the Grad-CAM image.
    """
    image_path=os.path.join(image_path,"test.jpg")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error clearing files in {output_folder}: {e}")
        
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    classes = ['Cancer detected', 'No cancer detected']  # Update classes to match your model's output
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = classes[predicted.item()]

    # Grad-CAM implementation
    target_class = predicted.item()
    target_layer_name = "layer4"  # Update to match your model architecture
    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    # Register hooks to save gradients and activations
    target_layer = dict([*model.named_modules()])[target_layer_name]
    target_layer.register_forward_hook(save_activation)
    target_layer.register_backward_hook(save_gradient)

    # Forward and backward pass
    outputs = model(input_tensor)
    model.zero_grad()
    class_score = outputs[0, target_class]
    class_score.backward()

    # Compute Grad-CAM
    grad = gradients[0].squeeze(0).detach().cpu().numpy()
    act = activations[0].squeeze(0).detach().cpu().numpy()
    weights = np.mean(grad, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam -= cam.min()
    cam /= cam.max()

    # Visualize Grad-CAM
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    gradcam_result = heatmap + np.float32(image) / 255
    gradcam_result = gradcam_result / gradcam_result.max()
    print("reached gradcam result")
    

    # Save Grad-CAM result as an image
    gradcam_image_path = os.path.join(output_folder, "result.jpg")
    result = cv2.imwrite(gradcam_image_path, np.uint8(255 * gradcam_result))
    if not result:
        print("Failed to write the Grad-CAM image.")
    print(gradcam_image_path)

    return predicted_class
