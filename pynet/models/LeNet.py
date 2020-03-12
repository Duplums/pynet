import torch
import torch.nn as nn

class LeNetLike(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        # ResNet-like 1st layer
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1)
        self.batchnorm2 = nn.BatchNorm3d(128)
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1)
        self.batchnorm3 = nn.BatchNorm3d(256)
        self.adaptive_maxpool = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(0.5)
        self.hidden_layer1 = nn.Linear(256, 256)
        self.hidden_layer2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)

        x = self.adaptive_maxpool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.hidden_layer1(x))
        x = self.dropout(x)
        x = self.hidden_layer2(x)

        return x.squeeze()


if __name__=="__main__":
    from pynet.datasets.core import DataManager
    from pynet.transforms import Crop, Padding, Normalize, LabelMapping
    from pynet.models.cam import plot_cam_heatmaps

    net = LeNetLike(in_channels=1, num_classes=2)
    pretrained_path = '/volatile/bd261576/checkpoints/scz_prediction/tmp/LeNet_Like_0_epoch_95.pth'
    chk = torch.load(pretrained_path)
    net.load_state_dict(chk['model'])
    stratif = {
        'train': {'study': ['BIOBD', 'SCHIZCONNECT-VIP', 'PRAGUE', 'HCP']},
        'test': {'study': 'BSNIP'}
    }
    data_manager = DataManager('/volatile/bd261576/Datasets/all_t1mri_mwp1_gs-raw_data32.npy',
                               '/volatile/bd261576/Datasets/all_t1mri_mwp1_participants.tsv',
                               batch_size=40,
                               labels=['diagnosis'],
                               sampler="weighted_random",
                               projection_labels={'diagnosis': ['control', 'schizophrenia']},
                               custom_stratification=stratif,
                               stratify_label='diagnosis',
                               input_transforms=[Crop((1, 121, 128, 121)), Padding((1, 128, 128, 128)),
                                                 Normalize(mean=0, std=1)],
                               stratify_label_transforms=[LabelMapping(control=0, schizophrenia=1)],
                               labels_transforms=[LabelMapping(control=0, schizophrenia=1)],
                               pin_memory=True,
                               drop_last=True)
    plot_cam_heatmaps("lenet", data_manager, {0: 'control', 1: 'schizophrenia'}, net)
