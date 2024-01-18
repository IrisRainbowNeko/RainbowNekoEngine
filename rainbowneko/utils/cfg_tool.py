from torch import nn


def change_num_classes(model, num_classes):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def block(target, *args, **kwargs):
    return dict(target=target, args=args, **kwargs)


def partial_block(target, *args, **kwargs):
    return dict(target=target, partial=True, args=args, **kwargs)
