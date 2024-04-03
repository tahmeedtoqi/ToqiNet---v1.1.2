import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
from src.ToqiNet import ToqiNet, ToqiDataset


# Define the transform to be applied to the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the saved model
if torch.cuda.is_available():
    checkpoint = torch.load('ToqiNet.weights')
else:
    checkpoint = torch.load('ToqiNet.weights', map_location=torch.device('cpu'))

class_to_idx = checkpoint['class_to_idx']
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
model = ToqiNet(num_classes=len(class_to_idx))  # Use the length of class_to_idx
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#  Here i have Move the model to the appropriate device
model.to(model.device)

# Load and preprocess the image
image_path = 'your path for test image'  # Update with the path to your image
image = Image.open(image_path).convert("RGB")
input_image = transform(image).unsqueeze(0)  # Add batch dimension

#  Here i have Move the input image to the appropriate device
input_image = input_image.to(model.device)
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

#  In this section the proggrame is performing inference
with torch.no_grad():
    output = model(input_image)
    probabilities = torch.softmax(output, dim=1)
    predicted_idx = torch.argmax(output).item()
    predicted_class = idx_to_class[predicted_idx]
    confidence = probabilities[0, predicted_idx].item() * 100

plt.imshow(image)
plt.axis('off')
plt.title(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%')
plt.show()
