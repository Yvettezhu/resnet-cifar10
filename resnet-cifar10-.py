import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247,  0.2435, 0.2616))])

train_transform = transforms.Compose(
    [transforms.RandomCrop(32, padding = 1),
     transforms.RandomHorizontalFlip(),

     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247,  0.2435, 0.2616))
    ]
    )


test_transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247,  0.2435, 0.2616))
    ]
    )

batch_size = 500

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform,
                                        download=False)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=test_transform
                                         )
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

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
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 10)

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
    model.train()



    # Decay learning rate
    #if (epoch+1) % 20 == 0:
        #curr_lr /= 10
        #update_lr(optimizer, curr_lr)

# Test the model


# Save the model checkpoint
#torch.save(model.state_dict(), 'resnet.ckpt')
