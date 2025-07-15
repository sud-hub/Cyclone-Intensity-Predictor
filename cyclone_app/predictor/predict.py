from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import io
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (once)
def load_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    conv_stem = model.features[0][0]
    new_conv = torch.nn.Conv2d(4, conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = conv_stem.weight
        new_conv.weight[:, 3:4, :, :] = conv_stem.weight[:, :1, :, :]
    model.features[0][0] = new_conv
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

def predict_intensity(image_file):
    image = Image.open(image_file).convert("RGB")  # Assuming RGB
    # Dummy raw channel (or fetch actual raw file from same filename)
    raw = image.convert("L")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    bt_tensor = transform(image)
    raw_tensor = transform(raw)[0].unsqueeze(0)

    input_tensor = torch.cat([bt_tensor, raw_tensor], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).item()

    return round(output, 2)
