import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowOptimize(nn.Module):
    def __init__(self, in_channels, out_channels, upbound_window=255.0):
        super(WindowOptimize, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.upbound_window = upbound_window
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, model_config):
        super(DenseNet, self).__init__()
        num_classes = model_config["classes"]
        self.model_config = model_config
        self.model = pretrainedmodels.__dict__["densenet201"](
            num_classes=1000, pretrained="imagenet"
        )
        if self.model_config.adj_model:
            conv0 = self.model.features.conv0
            self.model.features.conv0 = nn.Conv2d(
                in_channels=9,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias,
            )

            # copy pretrained weights
            self.model.features.conv0.weight.data[:, :3, :, :] = conv0.weight.data
            self.model.features.conv0.weight.data[:, 3:6, :, :] = conv0.weight.data
            self.model.features.conv0.weight.data[:, 6:9, :, :] = conv0.weight.data
        elif self.model_config.is_multi_channel:
            conv0 = self.model.features.conv0
            self.model.features.conv0 = nn.Conv2d(
                in_channels=6,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias,
            )

            # copy pretrained weights
            self.model.features.conv0.weight.data[:, :3, :, :] = conv0.weight.data
            self.model.features.conv0.weight.data[:, 3:6, :, :] = conv0.weight.data

        self.in_features = self.model.last_linear.in_features
        self.model.logits = self.logits

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm1d(self.in_features)
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.bn2 = nn.BatchNorm1d(self.in_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.in_features, self.in_features)
        self.bn3 = nn.BatchNorm1d(self.in_features)
        self.last_linear = nn.Linear(self.in_features, num_classes)

    def logits(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.cat((nn.AdaptiveAvgPool2d(1)(x), nn.AdaptiveMaxPool2d(1)(x)), dim=1)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        # x = F.dropout(x, p=0.25)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        # x = F.dropout(x, p=0.5)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        feature = self.bn3(x)
        return feature

    def features(self):
        return self.feature

    def forward(self, concat):
        """
        img: 6chのinput
        neg_img: 同じ実験、plateの'negative_control'統計image 6ch
        pos_img: 同じ実験、plateの'positive_control'統計image 6ch
        """
        concat = concat.permute(0, 3, 1, 2)
        out = self.model.features(concat)
        self.feature = self.model.logits(out)
        pred = self.last_linear(self.feature)
        return pred


class SeResNet(nn.Module):
    def __init__(self, model_config):
        super(SeResNet, self).__init__()
        num_classes = model_config["classes"]
        self.model_config = model_config
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](
            num_classes=1000, pretrained="imagenet"
        )
        if model_config.is_window:
            self.window_optimize_layer = WindowOptimize(in_channels=1, out_channels=3)
        if self.model_config.is_multi_channel:
            conv1 = self.model.layer0.conv1
            self.model.layer0.conv1 = nn.Conv2d(
                in_channels=6,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=conv1.bias,
            )

            # copy pretrained weights
            self.model.layer0.conv1.weight.data[:, :3, :, :] = conv1.weight.data
            self.model.layer0.conv1.weight.data[:, 3:6, :, :] = conv1.weight.data
        elif self.model_config.adj_model:
            conv1 = self.model.layer0.conv1
            self.model.layer0.conv1 = nn.Conv2d(
                in_channels=9,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=conv1.bias,
            )

            # copy pretrained weights
            self.model.layer0.conv1.weight.data[:, :3, :, :] = conv1.weight.data
            self.model.layer0.conv1.weight.data[:, 3:6, :, :] = conv1.weight.data
            self.model.layer0.conv1.weight.data[:, 6:9, :, :] = conv1.weight.data

        self.in_features = self.model.last_linear.in_features
        self.bn1 = nn.BatchNorm1d(self.in_features)
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.bn2 = nn.BatchNorm1d(self.in_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.in_features, self.in_features)
        self.bn3 = nn.BatchNorm1d(self.in_features)

        if model_config.branch_head:
            self.last_linear1 = nn.Linear(self.in_features, num_classes - 1)
            self.last_linear2 = nn.Linear(num_classes - 1, 1)
        else:
            self.last_linear = nn.Linear(self.in_features, num_classes)

    def logits(self, x):
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        # x = F.dropout(x, p=0.25)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        # x = F.dropout(x, p=0.25)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        x = self.bn3(x)

        return x

    def features(self):
        return self.feature

    def forward(self, concat):
        """
        img: 6chのinput
        neg_img: 同じ実験、plateの'negative_control'統計image 6ch
        pos_img: 同じ実験、plateの'positive_control'統計image 6ch
        """
        concat = concat.permute(0, 3, 1, 2)
        if self.model_config.is_window:
            out = self.window_optimize_layer(concat)
            out = self.model.features(out)
        else:
            out = self.model.features(concat)
        self.feature = self.logits(out)
        if self.model_config.branch_head:
            pred1 = self.last_linear1(self.feature)
            pred2 = self.last_linear2(pred1)
            pred = torch.cat([pred2, pred1], dim=-1)
        else:
            pred = self.last_linear(self.feature)

        return pred


def get_densenet(model_config):
    model = DenseNet(model_config)
    return model


def get_senet(model_config):
    model = SeResNet(model_config)
    return model


def get_model(model_name, **params):
    print("model name:", model_name)
    f = globals().get("get_" + model_name)
    return f(**params)
