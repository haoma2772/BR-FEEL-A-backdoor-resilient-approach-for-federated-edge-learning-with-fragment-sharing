import torch
import torch.nn as nn
import torchvision.models as models
from utility import load_config

# Define a five-layer MLP model
class MLP(nn.Module):
    def __init__(self, input_size, num_channels, num_classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation function in the last layer as cross-entropy loss will be used
        return x


class CustomCNN(nn.Module):
    class CustomCNN(nn.Module):
        def __init__(self, input_channels, num_classes, number_channels=1):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(16)  # Use adaptive average pooling to resize features to 16x16
            self.fc1 = nn.Linear(number_channels * 16 * 16, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)
            return x

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, input_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Assuming input_size is a tuple (height, width)
        fc_input_size = 128 * input_size // 64
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class SimpleCNNWithExtraConv(nn.Module):
    def __init__(self, input_channels, num_classes, input_size):
        super(SimpleCNNWithExtraConv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # New convolutional layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Assuming input_size is a tuple (height, width)
        fc_input_size = 256 * (input_size // 256)
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # New convolutional layer
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class Net(nn.Module):
    def __init__(self, config_list):
        super(Net, self).__init__()
        model_name, dataset_name = config_list['model']['name'], config_list['dataset']['name']
        if dataset_name == 'mnist':
            input_size = 224 * 224  # MNIST images are 28x28 pixels
            num_classes = 10  # MNIST has 10 classes
            number_channels = 3
        elif dataset_name in ['cifar10', 'cifar100']:
            input_size = 224 * 224 * 3
            # CIFAR-10 and CIFAR-100 images are 32x32 pixels with 3 channels
            number_channels = 3
            num_classes = 10 if dataset_name == 'cifar10' else 100
        elif dataset_name == 'imagenet1k':
            input_size = 224 * 224 * 3  # ImageNet images are 224x224 pixels with 3 channels
            num_classes = 1000  # ImageNet has 1000 classes
            number_channels = 3
        elif dataset_name == 'gtsrb':
            input_size = 3 * 224 * 224
            num_classes = 43  # GTSRB has 43 classes
            number_channels = 3

        if model_name == 'MLP':
            self.model = MLP(input_size=input_size, num_channels=number_channels, num_classes=num_classes)
        elif model_name == 'cnn':
            self.model = SimpleCNNWithExtraConv(input_channels=number_channels, 
                                                num_classes=num_classes, input_size=input_size)
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'shufflenet':
            self.model = models.shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
        elif model_name == 'mobilenetv2':
            self.model = models.mobilenet_v2(weights=None, num_classes=num_classes)
        elif model_name == 'alexnet':
            self.model = models.alexnet(weights=None, num_classes=num_classes)
        elif model_name == 'googlenet':
            self.model = models.googlenet(weights=None, num_classes=num_classes)
        elif model_name == 'densenet121':
            self.model = models.densenet121(weights=None, num_classes=num_classes)
        elif model_name == 'vision_transformer':
            self.model = models.vit_b_16(weights=None, num_classes=num_classes)
        elif model_name == 'vgg19':
            self.model = models.vgg19(weights=None, num_classes=num_classes)
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unsupported model: {}".format(model_name))

    def forward(self, x):
        return self.model(x)
