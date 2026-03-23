import torch.nn as nn
import torchvision.models as models


class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()

        self.model = models.densenet121(pretrained=pretrained)
        
        # Replace the classifier head for binary output (same as CNN's num_classes=1)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)