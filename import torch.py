import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from pathlib import Path
from torchvision import datasets


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
image_path = Path('gelsight/')
train_path = Path('split/train')
test_path = Path('split/test')
'''# Set seed
random.seed(20) # <- try changing this and see what happens
# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*.png"))
# 2. Get random image path
random_image_path = random.choice(image_path_list)
# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem
# 4. Open image
img = Image.open(random_image_path)
# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
img_as_array = np.asarray(img)

plot1 = plt.subplot2grid((1, 2), (0, 0))
plot2 = plt.subplot2grid((1, 2), (0, 1))

# Plot the image with matplotlib
plot1.plot(figsize=(10, 7))
plot1.imshow(img_as_array)

'''

data_transform = transforms.Compose([
    transforms.CenterCrop(480),
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
'''

with Image.open(random_image_path) as f:
    transformed_image = data_transform(f).permute(1, 2, 0)
t_img_as_array = np.asarray(transformed_image)
plot2.imshow(t_img_as_array)
plt.show()
'''
data = datasets.ImageFolder(root = image_path, 
                            transform = data_transform)

#print(data)
#train_size = int(0.7*len(data))
#test_size = len(data) - train_size
#train_dataset, test_dataset = torch.utils.data.random_split(data,[train_size, test_size])
#print(train_dataset)
#class_names = data.class_to_idx
#print(class_names)
#img, label = train_dataset[0][0],train_dataset[0][1]
#print(img)
#print(img.shape)
#print(img.dtype)
#print(label)
#print(type(label))  

splitfolders.ratio('gelsight/', output="split1",
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
splitfolders.ratio('gelsight/', output="split1",
    seed=1337, fixed=(500, 100), oversample=False, group_prefix=None, move=False)
import os
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"there are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

#walk_through_dir("split")

#train_data = datasets.ImageFolder(root = train_path, 
                                  #transform = data_transform)
#test_data = datasets.ImageFolder(root = test_path, 
                                  #transform = data_transform)

#train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                #batch_size = 500,
                                #num_workers = 1,
                                #shuffle = True)

#test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                #batch_size = 500,
                                #num_workers = 1,
                                #shuffle = True)

'''
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)
def batchnorm( out_channels):
    return nn.BatchNorm2d( out_channels)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        #self.in_channels = 32
        self.conv = conv3x3(3, 64)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            ResidualBlock(64,64,1,None),
            ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, 2, downsample= nn.Sequential (conv1x1(64,128,2),
                          nn.BatchNorm2d(128))),
            ResidualBlock(128,128))
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, 2, downsample = nn.Sequential (conv1x1(128,256,2),
                                                               nn.BatchNorm2d(256)) ),
            ResidualBlock(256, 256))
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, 2, downsample = nn.Sequential (conv1x1(256,512,2),
                                                               nn.BatchNorm2d(512)) ),
            ResidualBlock(512, 512))
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = ResNet().to(device)

num_epochs = 50
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate #,weight_decay= 1e-4
)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, epochs=50,
                                                steps_per_epoch=len(train_loader))

# For updating learning rate
#def update_lr(optimizer, lr):
    #for param_group in optimizer.param_groups:
        #param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #nn.utils.clip_grad_value_(model.parameters(), 0.1)

        # Backward and optimize
        #optimizer.step
        optimizer.zero_grad()
        loss.backward()
        #if epoch == 20 or epoch ==1:
            #plot_grad_flow()
        
    
        optimizer.step()
        sched.step()

        
    print ("Epoch [{}/{}]: {:.4f}"
    .format(epoch+1, num_epochs, loss.item()))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    model.train()'''