from torch import nn


def change_num_classes(model, num_classes):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
