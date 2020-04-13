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
from pynet.models.psynet import PsyNet
from pynet.models.alpha_wgan import *
from pynet.transforms import *
from pynet.utils import get_chk_name
import os
import pickle

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="/home_local/bd261576/all_t1mri_mwp1_gs-raw_data32_tocheck.npy")
    parser.add_argument("--metadata_path", type=str, default="/home_local/bd261576/all_t1mri_mwp1_participants.tsv")
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--with_test", action="store_true", help="Whether to test the model on the best validation point")
    parser.add_argument("--checkpoint_dir", type=str, default="/neurospin/psy_sbox/bd261576/checkpoints")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("--nb_folds", type=int, default=5)
    parser.add_argument("--pin_mem", type=bool, default=True)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--net", type=str)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--nb_epochs", type=int)
    parser.add_argument("--with_logit", action="store_true")
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

    # stratif = {
    #     'train': {'study': ['HCP', 'IXI'], 'diagnosis': 'control'},
    #     'validation': {'study': 'BIOBD', 'diagnosis': 'control'},
    #     'test': {'study': ['SCHIZCONNECT-VIP', 'BIOBD', 'BSNIP'], 'diagnosis': ['control', 'schizophrenia', 'FEP']}
    # }

    stratif = {
        'train': {'study': ['HCP', 'IXI', 'SCHIZCONNECT-VIP', 'PRAGUE'], 'diagnosis': ['control', 'schizophrenia']},
        'test': {'study': 'BSNIP', 'diagnosis': ['control', 'schizophrenia']}
    }

    add_to_input = None
    add_input = False
    labels=['diagnosis']
    sampler="random"
    projection_labels={'study': ['BSNIP', 'HCP', 'IXI', 'SCHIZCONNECT-VIP', 'PRAGUE'],
                       'diagnosis': ['control', 'schizophrenia', 'FEP']}
    stratify_label=None#"site"
    input_transforms=[Crop((1, 121, 128, 121)), Padding((1, 128, 128, 128)), Normalize()]
    strat_label_transforms=None#[LabelMapping(**{site: indice for (indice, site) in enumerate(sorted(set(df['site'])))})]
    labels_transforms=[LabelMapping(schizophrenia=1, control=0)]
    data_augmentation=[]#[RandomAffineTransform3d(40, 0.1)]
    output_transforms=None
    patch_size=None
    input_size=None
    metrics=None
    std_optim = True
    with_validation=True
    num_classes = 2

    if args.net == "resnet18":
        net = resnet18(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                       with_dropout=False)
    if args.net == "resnet34":
        net = resnet34(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                       with_dropout=False)
    elif args.net == "resnet50":
        net = resnet50(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                       with_dropout=False)
    elif args.net == "resnext50":
        net = resnext50_32x4d(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                              with_dropout=False)
    elif args.net == "resnet101":
        net = resnet101(pretrained=False, num_classes=num_classes, in_channels=1, with_grid_attention=False,
                        with_dropout=False)
    elif args.net == "vgg11":
        net = vgg11(in_channels=1, num_classes=num_classes,  init_weights=True, dim="3d",
                    with_grid_attention=False, batchnorm=True)
    elif args.net == "vgg16":
        net = vgg16(in_channels=1, num_classes=num_classes,  init_weights=True, dim="3d",
                    with_grid_attention=False, batchnorm=True)
    elif args.net == "densenet121":
        net = densenet121(progress=False, num_classes=num_classes)
    elif args.net == "alpha_wgan":
        f = os.path.join(os.path.dirname(inputs_path), "data_alpha-wGAN_encoded.npy")
        net = Alpha_WGAN(lr=args.lr, device=('cuda' if args.cuda else 'cpu'), path_to_file=f)
        std_optim = False
    elif args.net == "psy_net":
        f = os.path.join(os.path.dirname(inputs_path), "data_psynet.npy")
        net = PsyNet(num_classes=num_classes, path_to_file=f)
        std_optim = False
    else:
        raise ValueError('Unknown network %s' % args.net)

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

    #loss = MultiTaskLoss([nn.L1Loss(), nn.BCEWithLogitsLoss()], [0.1, 1])
    loss = nn.BCEWithLogitsLoss()
    # loss = net.get_loss
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-5)

    model = Base(model=net, loss=loss,
                 metrics=metrics,
                 optimizer=optim,
                 pretrained = args.pretrained_path,
                 use_cuda=args.cuda)

    scheduler = None#torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.9)

    if args.net == "alpha_wgan" and args.test_ssim:
        ms_ssim_rec, ms_ssim_true = net.compute_MS_SSIM(manager1.get_dataloader(test=True).test,
                                                        batch_size=batch_size, save_img=True,
                                                        saving_dir=os.path.dirname(inputs_path))
        print("MS-SSIM on true samples: %f, rec_samples: %f"%
              (ms_ssim_true, ms_ssim_rec), flush=True)
        ms_ssim_rand = net.compute_MS_SSIM(N=400, batch_size=batch_size, save_img=True,
                                           saving_dir=os.path.dirname(inputs_path))
        print("MS-SSIM on 200 random samples: %f"%ms_ssim_rand, flush=True)
        exit(0)

    if not args.test:
        train_history, valid_history = model.training(manager1,
                                                      nb_epochs=args.nb_epochs,
                                                      scheduler=scheduler,
                                                      with_validation=with_validation,
                                                      checkpointdir=args.checkpoint_dir,
                                                      nb_epochs_per_saving=1,
                                                      exp_name=args.exp_name,
                                                      with_visualization=True,
                                                      standard_optim=std_optim)

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
        y, X, y_true, loss, values = model.testing(manager1, with_visuals=False, with_logit=args.with_logit,
                                                   saving_dir=args.checkpoint_dir,
                                                   exp_name=exp_name, standard_optim=std_optim)

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






