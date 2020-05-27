import pandas as pd
import argparse
from pynet.core import Base
from pynet.datasets.core import DataManager
from pynet.losses import *
from pynet.models.pinayanet import SemiSupervisedAE
from pynet.models.resnet import *
from pynet.models.densenet import *
from pynet.models.vgg_grid_attention import *
from pynet.models.colenet import ColeNet
from pynet.models.DIDD import *
from pynet.models.psynet import PsyNet
from pynet.models.alpha_wgan import *
from pynet.transforms import *
from pynet.utils import get_chk_name
import os

def build_network(name, num_classes):
    if name == "resnet18":
        net = resnet18(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                       with_dropout=False)
    elif name == "resnet34":
        net = resnet34(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                       with_dropout=False, prediction_bias=False)
    elif name == "light_resnet34":
        net = resnet34(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                       with_dropout=False, prediction_bias=False, initial_kernel_size=3)
    elif name == "resnet50":
        net = resnet50(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                       with_dropout=False)
    elif name == "resnext50":
        net = resnext50_32x4d(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                              with_dropout=False)
    elif name == "resnet101":
        net = resnet101(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                        with_dropout=False)
    elif name == "vgg11":
        net = vgg11(in_channels=1, num_classes=num_classes,  init_weights=True, dim="3d",
                    with_grid_attention=False, batchnorm=True)
    elif name == "vgg16":
        net = vgg16(in_channels=1, num_classes=num_classes,  init_weights=True, dim="3d",
                    with_grid_attention=False, batchnorm=True)
    elif name == "densenet121":
        net = densenet121(progress=False, num_classes=num_classes)
    elif name== 'cole_net':
        net = ColeNet(num_classes, [1, 128, 128, 128])
    elif name == "alpha_wgan":
        net = Alpha_WGAN(lr=args.lr, device=('cuda' if args.cuda else 'cpu'), use_kl=True, path_to_file=None)
    elif name == "alpha_wgan_predictors":
        net = Alpha_WGAN_Predictors(latent_dim=1000)
    elif name == "psy_net":
        alpha_wgan = Alpha_WGAN(lr=args.lr, device=('cuda' if args.cuda else 'cpu'), use_kl=True)
        net = PsyNet(alpha_wgan, num_classes=num_classes, lr=args.lr, device=('cuda' if args.cuda else 'cpu'))
    else:
        raise ValueError('Unknown network %s' % name)

    ## Networks to come...
    # net = UNet(1, in_channels=1, depth=4, merge_mode=None, batchnorm=True,
    #            skip_connections=False, down_mode="maxpool", up_mode="upsample",
    #            mode="classif", input_size=(1, 128, 144, 128), freeze_encoder=True,
    #            nb_regressors=1)
    # net = LeNetLike(in_channels=1, num_classes=2)
    # net = CoRA(1, 1)
    # net = DIDD(in_channels=1, sep=25, input_size=[1, 128, 128, 128], device='cuda')
    # net = SchizNet(1, [1, 128, 128, 128], batch_size)

    return net


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="/home_local/bd261576/all_t1mri_mwp1_gs-raw_data32_tocheck.npy")
    parser.add_argument("--metadata_path", type=str, default="/home_local/bd261576/all_t1mri_mwp1_participants.tsv")
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--with_test", action="store_true", help="Whether to test the model on the best validation point")
    parser.add_argument("--checkpoint_dir", type=str, default="/neurospin/psy_sbox/bd261576/checkpoints")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--N_train_max", type=int, default=None)
    parser.add_argument("--nb_epochs_per_saving", type=int, default=5)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("--nb_folds", type=int, default=5)
    parser.add_argument("--pin_mem", type=bool, default=True)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--db", choices=["healthy", "scz_kfolds", "tiny_scz_kfolds"], required=True)
    parser.add_argument("--sampler", choices=["random", "weighted_random"], required=True)
    parser.add_argument("--labels", nargs='+', type=str, help="Label(s) to be predicted")
    parser.add_argument("--stratify_label", type=str, help="Label used for the stratification of the train/val split")
    parser.add_argument("--metrics", nargs='+', type=str, help="Metrics to be computed at each epoch")
    parser.add_argument("--add_input", action="store_true", help="Whether to add the input data to the output "
                                                                 "(e.g for a reconstruction task)")
    parser.add_argument("--test_all_epochs", action="store_true")
    parser.add_argument("--net", type=str, help="Initial learning rate")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--nb_epochs", type=int)
    parser.add_argument("--load_optimizer", action="store_true", help="If <pretrained_path> is set, loads also the "
                                                                      "optimizer's weigth")
    parser.add_argument("--with_logit", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--test_ssim", action="store_true", help="Specific to AlphaWGAN net")
    args = parser.parse_args()

    inputs_path = args.input_path
    metadata_path = args.metadata_path
    df = pd.read_csv(args.metadata_path, sep='\t')
    batch_size = args.batch_size
    nb_folds = args.nb_folds
    pin_memory = args.pin_mem
    drop_last = args.drop_last
    labels = args.labels or []
    stratify_label = args.stratify_label
    sampler=args.sampler
    N_train_max = args.N_train_max
    metrics = args.metrics
    add_input = args.add_input
    db = args.db

    projection_labels = {
        'diagnosis': ['control', 'FEP', 'schizophrenia']
    }

    if db == "healthy":
        stratif = {
            'train': {'study': ['HCP', 'IXI'], 'diagnosis': 'control'},
            'validation': {'study': 'BIOBD', 'diagnosis': 'control'},
            'test': {'study': ['BSNIP'], 'diagnosis': ['control']}
        }
    elif db == "scz_kfolds":
        stratif = {
            'train': {'study': ['HCP', 'IXI', 'SCHIZCONNECT-VIP', 'PRAGUE'],
                      'diagnosis': ['control', 'FEP', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']}
        }
    elif db == "tiny_scz_kfolds":
        stratif = {
            'train': {'study': ['SCHIZCONNECT-VIP', 'PRAGUE'],
                      'diagnosis': ['control', 'FEP', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']}
        }

    if args.net == "alpha_wgan":
        input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'), HardNormalization()]
    else:
        input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'), Normalize()]

    with_validation = (nb_folds > 1) or ('validation' in stratif)
    # <label>: [LabelMapping(), IsCategorical]
    known_labels = {'age': [LabelMapping(), False],
                    'sex': [LabelMapping(), True],
                    'site':[LabelMapping(**{site: indice for (indice, site) in enumerate(sorted(set(df['site'])))}), True],
                    'diagnosis': [LabelMapping(schizophrenia=1, FEP=1, control=0), True]
                    }

    assert set(labels) <= set(known_labels.keys()), \
        "Unknown label(s), chose from {}".format(set(known_labels.keys()))

    assert (stratify_label is None) or (stratify_label in set(known_labels.keys())), \
        "Unknown stratification label, chose from {}".format(set(known_labels.keys()))

    strat_label_transforms = [known_labels[stratify_label][0]] \
        if (stratify_label is not None and known_labels[stratify_label][0] is not None) else None
    categorical_strat_label = known_labels[stratify_label][1] if stratify_label is not None else None
    if len(labels) == 0:
        labels_transforms = None
    elif len(labels) == 1:
        labels_transforms = [known_labels[labels[0]][0]]
    else:
        labels_transforms = [lambda in_labels: [known_labels[labels[i]][0](l) for i, l in enumerate(in_labels)]]

    add_to_input = None
    data_augmentation=[RandomFlip(hflip=True, vflip=True)]
    self_supervision=None#RandomPatchInversion(patch_size=15, data_threshold=0)
    output_transforms=None
    patch_size=None
    input_size=None
    std_optim = (args.net not in  ["alpha_wgan", "psy_net"])
    num_classes = 1

    net = build_network(args.net, num_classes)

    # loss = SSIM()
    # loss = SAE_Loss(rho=0.05, n_hidden=400, lambda_=0.1, device="cuda")
    # loss = [net.zeros_rec_adv_loss, net.disc_loss]
    # weight = torch.tensor([365./2045, 1.0]) # 365 SCZ (pos) / 2045 CTL (neg)
    # weight = weight.to('cuda')
    # loss = nn.CrossEntropyLoss(weight=weight)
    #loss = nn.L1Loss()
    if list(labels) == ["age"]:
        loss = nn.L1Loss()
    elif list(labels) == ["sex"]:
        loss = nn.BCEWithLogitsLoss()
    elif list(labels) == ["diagnosis"]:
        loss = nn.BCEWithLogitsLoss()

    manager1 = DataManager(inputs_path, metadata_path,
                           batch_size=batch_size,
                           number_of_folds=nb_folds,
                           add_to_input=add_to_input,
                           add_input=add_input,
                           labels=labels,
                           sampler=sampler,
                           projection_labels=projection_labels,
                           custom_stratification=stratif,
                           categorical_strat_label=categorical_strat_label,
                           stratify_label=stratify_label,
                           N_train_max=N_train_max,
                           input_transforms=input_transforms,
                           stratify_label_transforms=strat_label_transforms,
                           labels_transforms=labels_transforms,
                           data_augmentation=data_augmentation,
                           self_supervision=self_supervision,
                           output_transforms=output_transforms,
                           patch_size=patch_size,
                           input_size=input_size,
                           pin_memory=pin_memory,
                           drop_last=drop_last,
                           device=('cuda' if args.cuda else 'cpu'))

    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-5)


    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.9)

    if args.net == "alpha_wgan" and args.test_ssim:
        net.load_state_dict(torch.load(args.pretrained_path)["model"], strict=True)
        net = net.to(('cuda' if args.cuda else 'cpu'))
        ms_ssim_rec, ms_ssim_true, ms_ssim_inter_rec = net.compute_MS_SSIM(manager1.get_dataloader(test=True).test,
                                                        batch_size=batch_size, save_img=True,
                                                        saving_dir=args.checkpoint_dir)
        print("MS-SSIM on true samples: %f, rec_samples: %f, inter-rec samples: %f"%
              (ms_ssim_true, ms_ssim_rec, ms_ssim_inter_rec), flush=True)
        ms_ssim_rand = net.compute_MS_SSIM(N=400, batch_size=batch_size, save_img=True,
                                           saving_dir=args.checkpoint_dir)
        print("MS-SSIM on 400 random samples: %f"%ms_ssim_rand, flush=True)
        exit(0)

    if not args.test:
        model = Base(model=net, loss=loss,
                     metrics=metrics,
                     optimizer=optim,
                     pretrained=args.pretrained_path,
                     load_optimizer=args.load_optimizer,
                     use_cuda=args.cuda)
        train_history, valid_history = model.training(manager1,
                                                      nb_epochs=args.nb_epochs,
                                                      scheduler=scheduler,
                                                      with_validation=with_validation,
                                                      checkpointdir=args.checkpoint_dir,
                                                      nb_epochs_per_saving=args.nb_epochs_per_saving,
                                                      exp_name=args.exp_name,
                                                      with_visualization=True,
                                                      standard_optim=std_optim)

    if args.with_test or args.test:
        # Get the last point and test it, for each fold
        if args.test_all_epochs:
            epochs_tested = list(range(args.nb_epochs_per_saving, args.nb_epochs, args.nb_epochs_per_saving))+[args.nb_epochs-1]
        else:
            epochs_tested = [args.nb_epochs-1]
        for fold in range(manager1.number_of_folds):
            for epoch in epochs_tested:
                pretrained_path = os.path.join(args.checkpoint_dir, get_chk_name(args.exp_name, fold, epoch))
                exp_name = "Test_" + args.exp_name + "_fold{}_epoch{}".format(fold, epoch)
                model = Base(model=net, loss=loss,
                             metrics=metrics,
                             optimizer=optim,
                             pretrained=pretrained_path,
                             use_cuda=args.cuda)
                model.testing(manager1, with_visuals=False, with_logit=args.with_logit,
                              predict=args.predict,
                              saving_dir=args.checkpoint_dir,
                              exp_name=exp_name, standard_optim=std_optim)






