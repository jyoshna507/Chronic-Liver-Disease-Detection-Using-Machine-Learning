import torch
from app.models.capsule_resnet import Model1
from app.models.dedswin import Model2

try:
    print("Checking CUDA availability...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    x = torch.randn(2, 1, 256, 256).to(device)

    model1 = Model1().to(device)
    seg1, cls1 = model1(x)
    print("Model1 OK:", seg1.shape, cls1.shape)

    model2 = Model2().to(device)
    seg2, cls2 = model2(x)
    print("Model2 OK:", seg2.shape, cls2.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
