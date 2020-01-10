import torch
import torch.onnx
import torchvision.models
import torchvision.transforms as transforms
# A model class instance (class not shown)
model = torchvision.models.resnet50()
model.fc = torch.nn.Linear(in_features=2048, out_features=1)
# Load the weights from a file (.pth usually)

# Load the weights now into a model net architecture defined by our class
model.load_state_dict(torch.load('model/model-resnet50.pth', map_location='cpu')) 

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(1, 3, 224, 224)

#to onn
torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")

#to mlmodel
mlmodel = convert(model='onnx_model_name.onnx')
mlmodel.save('ios.mlmodel')