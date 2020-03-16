import pandas as pd
import argparse
from pynet.core import Base
from pynet.datasets.core import DataManager
from pynet.losses import *
from pynet.models.pinayanet import SemiSupervisedAE
from pynet.models.resnet import *
from pynet.models.densenet import *
from pynet.models.vgg_grid_attention import *
from pynet.models.DIDD import *
from pynet.transforms import *
from pynet.utils import get_chk_name
import os
import pickle

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/all_t1mri_mwp1_gs-raw_data32_tocheck.npy")
    parser.add_argument("--metadata_path", type=str, default="/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/all_t1mri_mwp1_participants.tsv")
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--with_test", action="store_true", help="Whether to test the model on the best validation point")
    parser.add_argument("--checkpoint_dir", type=str, default="/neurospin/psy_sbox/bd261576/checkpoints")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("--nb_folds", type=int, default=5)
    parser.add_argument("--pin_mem", type=bool, default=True)
    parser.add_argument("--drop_last", type=bool, default=False)
    parser.add_argument("--net", type=str)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--nb_epochs", type=int)
    parser.add_argument("--cuda", type=bool, default=True)

    args = parser.parse_args()

    inputs_path = args.input_path
    metadata_path = args.metadata_path
    df = pd.read_csv(args.metadata_path, sep='\t')
    batch_size = args.batch_size
    nb_folds = args.nb_folds
    pin_memory = args.pin_mem
    drop_last = args.drop_last

    stratif = {
        'train': {'study': ['HCP', 'IXI']},
        'validation': {'study': 'BIOBD'},
        'test': {'study': 'BSNIP'}
    }

    add_to_input = None
    add_input = False
    labels=["age"]
    sampler="random"
    projection_labels={'diagnosis': 'schizophrenia'}
    stratify_label=None
    input_transforms=[Crop((1, 121, 128, 121)), Padding((1, 128, 128, 128)), Normalize(mean=0, std=1)]
    strat_label_transforms=None #LabelMapping( **{site: indice for (indice, site) in enumerate(sorted(set(df['site'])))}
    labels_transforms=None
    data_augmentation=None #[GaussianNoise(0.1), RandomAffineTransform3d(30, 0.1)]
    output_transforms=None
    patch_size=None
    input_size=None
    metrics=None

    if args.net == "resnet18":
        net = resnet18(pretrained=False, num_classes=1, in_channels=1, with_grid_attention=False,
                       with_dropout=False)
    if args.net == "resnet34":
        net = resnet34(pretrained=False, num_classes=1, in_channels=1, with_grid_attention=False,
                       with_dropout=False)
    elif args.net == "resnet50":
        net = resnet50(pretrained=False, num_classes=1, in_channels=1, with_grid_attention=False,
                       with_dropout=False)
    elif args.net == "resnext50":
        net = resnext50_32x4d(pretrained=False, num_classes=1, in_channels=1, with_grid_attention=False,
                              with_dropout=False)
    elif args.net == "resnet101":
        net = resnet101(pretrained=False, num_classes=1, in_channels=1, with_grid_attention=False,
                        with_dropout=False)
    elif args.net == "vgg11":
        net = vgg11(in_channels=1, num_classes=1,  init_weights=True, dim="3d",
                    with_grid_attention=False, batchnorm=True)
    elif args.net == "vgg16":
        net = vgg16(in_channels=1, num_classes=1,  init_weights=True, dim="3d",
                    with_grid_attention=False, batchnorm=True)
    elif args.net == "densenet121":
        net = densenet121(progress=False, num_classes=1)


    # net = UNet(1, in_channels=1, depth=4, merge_mode=None, batchnorm=True,
    #            skip_connections=False, down_mode="maxpool", up_mode="upsample",
    #            mode="classif", input_size=(1, 128, 144, 128), freeze_encoder=True,
    #            nb_regressors=1)

    # net = ColeNet(1, [1, 121, 145, 121])

    # net = LeNetLike(in_channels=1, num_classes=2)

    # net = CoRA(1, 1)

    # net = DIDD(in_channels=1, sep=25, input_size=[1, 128, 128, 128], device='cuda')

    # net = DenseNet(growth_rate=32, block_config=(6, 12, 12, 16), num_classes=1)
    # net = SchizNet(1, [1, 128, 128, 128], batch_size)


    manager1 = DataManager(inputs_path, metadata_path,
                           batch_size=batch_size,
                           number_of_folds=nb_folds,
                           add_to_input=add_to_input,
                           add_input=add_input,
                           labels=labels,
                           sampler=sampler,
                           projection_labels=projection_labels,
                           custom_stratification=stratif,
                           stratify_label=stratify_label,
                           input_transforms=input_transforms,
                           stratify_label_transforms=strat_label_transforms,
                           labels_transforms=labels_transforms,
                           data_augmentation=data_augmentation,
                           output_transforms=output_transforms,
                           patch_size=patch_size,
                           input_size=input_size,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

    # weight = torch.tensor([0.5, 1])
    # weight = weight.to('cuda')
    # loss = SSIM()
    # loss = SAE_Loss(rho=0.05, n_hidden=400, lambda_=0.1, device="cuda")
    # loss = [net.zeros_rec_adv_loss, net.disc_loss]
    loss = nn.L1Loss()

    # loss = net.get_loss
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-5)

    model = Base(model=net, loss=loss,
                 metrics=metrics,
                 optimizer=optim,
                 pretrained = args.pretrained_path,
                 use_cuda=args.cuda)

    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.97)

    if not args.test:
        train_history, valid_history = model.training(manager1,
                                                      nb_epochs=args.nb_epochs,
                                                      scheduler=scheduler,
                                                      checkpointdir=args.checkpoint_dir,
                                                      nb_epochs_per_saving=1,
                                                      exp_name=args.exp_name,
                                                      with_visualization=True)

    if args.with_test:
        if valid_history is not None:
            # Get the best validation point
            best_point = np.argmin(valid_history['validation_loss'][1])
            (best_fold, best_epoch) = valid_history['validation_loss'][0][best_point]
            pretrained_path = os.path.join(args.checkpoint_dir, get_chk_name(args.exp_name, best_fold, best_epoch))
            model = Base(model=net, loss=loss,
                         metrics=metrics,
                         optimizer=optim,
                         pretrained=pretrained_path,
                         use_cuda=args.cuda)
        else:
            raise ValueError("No Validation History found. Impossible to test the model")

    if args.test or args.with_test:
        if args.with_test:
            exp_name = "Test_"+args.exp_name+"_fold{}_epoch{}".format(best_fold, best_epoch)
        else:
            exp_name = args.exp_name
        y, X, y_true, loss, values = model.testing(manager1, with_visuals=False, saving_dir=args.checkpoint_dir,
                                                   exp_name=exp_name)

        # plot_cam_heatmaps("resnet18", manager1, {0: 'control', 1: 'schizophrenia'}, net)

    # manager2 = DataManager(inputs_path, metadata_path,
    #                        batch_size=batch_size,
    #                        number_of_folds=5,
    #                        add_input=False,
    #                        labels=["diagnosis"],
    #                        sampler="random",
    #                        projection_labels={'diagnosis': ['control', 'schizophrenia']},
    #                        custom_stratification=stratif,
    #                        stratify_label='diagnosis',
    #                        input_transforms=[Crop((1, 121, 128, 121)), Padding((1, 128, 128, 128)),
    #                                          Normalize(mean=0, std=1)],
    #                        stratify_label_transforms=[LabelMapping(control=0, schizophrenia=1)],
    #                        labels_transforms=[LabelMapping(control=0, schizophrenia=1)],
    #                        output_transforms=[Crop((1, 121, 128, 121)), Padding((1, 128, 128, 128)),
    #                                           Normalize(mean=0, std=1)],
    #                        pin_memory=True,
    #                        drop_last=True)




########################################################################################################################
## Domain Intersection Domain Difference (DIDD) Specific visualization --> To Move
        # batch_ctl = list(manager_ctl.get_dataloader(test=True).test)
        # batch_scz = list(manager_scz.get_dataloader(test=True).test)
        #
        # out = net(batch_ctl[0].inputs.to('cuda'), batch_scz[0].inputs.to('cuda'))
        #
        # from pynet.plotting.image import plot_data
        #
        # # 1st CONTROL+SCZ DECODED
        #
        # plot_data(net.ctl_decoded.detach().cpu().numpy(), nb_samples=10, random=False)
        #
        # plot_data(net.dx_decoded.detach().cpu().numpy())
        #
        # ctl_enc_dx_decoded = net.decoder(torch.cat([net.ctl_common,
        #                                             torch.zeros(net.ctl_spec.shape, device='cuda'),
        #                                             net.ctl_enc_dx], dim=1))
        # plot_data(ctl_enc_dx_decoded.detach().cpu().numpy(), nb_samples=8, random=False)
        #
        # ctl_spec = net.decoder(torch.cat([torch.zeros(net.ctl_common.shape, device='cuda'),
        #                                   net.ctl_spec, net.zero_encoding], dim=1))
        #
        # common_spec = net.decoder(torch.cat([net.ctl_common,
        #                                      torch.zeros(net.ctl_spec.shape, device='cuda'),
        #                                      net.zero_encoding], dim=1))
        #
        #
        #
        # plot_data(ctl_spec.detach().cpu().numpy())
        # plot_data(common_spec.detach().cpu().numpy())






