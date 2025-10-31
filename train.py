from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

dataset = datasets.ImageFolder(root='archive',transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset)-train_size
train_dataset,val_dataset = random_split(dataset,[train_size,val_size])
train_data = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
val_data = DataLoader(dataset=val_dataset,batch_size=64)

