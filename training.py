from pynet.core import Base
from pynet.datasets.core import DataManager
from pynet.models.resnet import *
from pynet.models.densenet import *
from pynet.models.vgg import *
from pynet.losses import *
from pynet.models.colenet import ColeNet
from pynet.models.psynet import PsyNet
from pynet.models.alpha_wgan import *
import pandas as pd
from json_config import CONFIG
from pynet.transforms import *


class BaseTrainer():

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.num_classes, args, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args, input_transforms=
        [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),
         Normalize()])
        self.loss = BaseTrainer.build_loss(args.loss, net=self.net)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, **CONFIG['optimizer']['Adam'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=args.gamma_scheduler,
                                                         **CONFIG['scheduler']['StepLR'])

        self.model = Base(model=self.net,
                          metrics=args.metrics,
                          pretrained=args.pretrained_path,
                          load_optimizer=args.load_optimizer,
                          use_cuda=args.cuda,
                          loss=self.loss,
                          optimizer=self.optimizer)

    def run(self):
        with_validation = (self.args.nb_folds > 1) or ('validation' in CONFIG['db'][self.args.db])
        train_history, valid_history = self.model.training(self.manager,
                                                           nb_epochs=self.args.nb_epochs,
                                                           scheduler=self.scheduler,
                                                           with_validation=with_validation,
                                                           checkpointdir=self.args.checkpoint_dir,
                                                           nb_epochs_per_saving=self.args.nb_epochs_per_saving,
                                                           exp_name=self.args.exp_name,
                                                           fold_index=self.args.folds,
                                                           standard_optim=getattr(self.net, 'std_optim', True),
                                                           with_visualization=False)

        return train_history, valid_history

    @staticmethod
    def build_loss(name, net=None):
        if name == 'l1':
            loss = nn.L1Loss()
        elif name == 'BCE':
            loss = nn.BCEWithLogitsLoss()
        elif name == 'BCE_concrete_dropout':
            assert net is not None, "A model is mandatory to compute the regularization term"
            loss = ConcreteDropoutLoss(net, nn.BCEWithLogitsLoss(), weight_regularizer=1e-6, dropout_regularizer=1e-5)
        else:
            raise ValueError("Loss not yet implemented")
            # loss = SSIM()
            # loss = SAE_Loss(rho=0.05, n_hidden=400, lambda_=0.1, device="cuda")
            # loss = [net.zeros_rec_adv_loss, net.disc_loss]
            # weight = torch.tensor([365./2045, 1.0]) # 365 SCZ (pos) / 2045 CTL (neg)
            # weight = weight.to('cuda')
            # loss = nn.CrossEntropyLoss(weight=weight)
        return loss

    @staticmethod
    def build_network(name, num_classes, args, **kwargs):
        if name == "resnet18":
            net = resnet18(pretrained=False, num_classes=num_classes, **kwargs)
        elif name == "resnet34":
            net = resnet34(pretrained=False, num_classes=num_classes, prediction_bias=False, **kwargs)
        elif name == "light_resnet34":
            net = resnet34(pretrained=False, num_classes=num_classes, initial_kernel_size=3, **kwargs)
        elif name == "resnet50":
            net = resnet50(pretrained=False, num_classes=num_classes, **kwargs)
        elif name == "resnext50":
            net = resnext50_32x4d(pretrained=False, num_classes=num_classes, **kwargs)
        elif name == "resnet101":
            net = resnet101(pretrained=False, num_classes=num_classes, **kwargs)
        elif name == "vgg11":
            net = vgg11(num_classes=num_classes, init_weights=True, dim="3d", **kwargs)
        elif name == "vgg16":
            net = vgg16(num_classes=num_classes, init_weights=True, dim="3d", **kwargs)
        elif name == "densenet121":
            net = densenet121(progress=False, num_classes=num_classes, drop_rate=args.dropout, bayesian=args.bayesian,
                              concrete_dropout=args.concrete_dropout, **kwargs)
        elif name in ["densenet121_block%i"%i for i in range(1,5)]:
            block = name[-6:]
            net = densenet121(progress=False, num_classes=num_classes, drop_rate=args.dropout, bayesian=args.bayesian,
                              concrete_dropout=args.concrete_dropout, out_block=block, **kwargs)
        elif name == 'cole_net':
            net = ColeNet(num_classes, [1, 128, 128, 128], concrete_dropout=args.concrete_dropout)
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
            # net = SchizNet(1, [1, 128, 128, 128], batch_size)

        return net

    @staticmethod
    def build_data_manager(args, **kwargs):
        df = pd.read_csv(args.metadata_path, sep='\t')
        labels = args.labels or []
        add_to_input = None
        data_augmentation = [RandomFlip(hflip=True, vflip=True)]
        self_supervision = None  # RandomPatchInversion(patch_size=15, data_threshold=0)
        input_transforms = kwargs.get('input_transforms')
        output_transforms = None
        patch_size = None
        input_size = None

        projection_labels = {
            'diagnosis': ['control', 'FEP', 'schizophrenia']
        }

        stratif = CONFIG['db'][args.db]


        ## Set the preprocessing step with an exception for GAN
        if input_transforms is None:
            if args.net == "alpha_wgan":
                input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),
                                    HardNormalization()]
            else:
                input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'), Normalize()]

        ## Set the basic mapping between a label and an integer

        # <label>: [LabelMapping(), IsCategorical]
        known_labels = {'age': [LabelMapping(), False],
                        'sex': [LabelMapping(), True],
                        'site': [
                            LabelMapping(**{site: indice for (indice, site) in enumerate(sorted(set(df['site'])))}),
                            True],
                        'diagnosis': [LabelMapping(schizophrenia=1, FEP=1, control=0), True]
                        }

        assert set(labels) <= set(known_labels.keys()), \
            "Unknown label(s), chose from {}".format(set(known_labels.keys()))

        assert (args.stratify_label is None) or (args.stratify_label in set(known_labels.keys())), \
            "Unknown stratification label, chose from {}".format(set(known_labels.keys()))

        strat_label_transforms = [known_labels[args.stratify_label][0]] \
            if (args.stratify_label is not None and known_labels[args.stratify_label][0] is not None) else None
        categorical_strat_label = known_labels[args.stratify_label][1] if args.stratify_label is not None else None
        if len(labels) == 0:
            labels_transforms = None
        elif len(labels) == 1:
            labels_transforms = [known_labels[labels[0]][0]]
        else:
            labels_transforms = [lambda in_labels: [known_labels[labels[i]][0](l) for i, l in enumerate(in_labels)]]



        manager = DataManager(args.input_path, args.metadata_path,
                               batch_size=args.batch_size,
                               number_of_folds=args.nb_folds,
                               add_to_input=add_to_input,
                               add_input=args.add_input,
                               labels=labels,
                               sampler=args.sampler,
                               projection_labels=projection_labels,
                               custom_stratification=stratif,
                               categorical_strat_label=categorical_strat_label,
                               stratify_label=args.stratify_label,
                               N_train_max=args.N_train_max,
                               input_transforms=input_transforms,
                               stratify_label_transforms=strat_label_transforms,
                               labels_transforms=labels_transforms,
                               data_augmentation=data_augmentation,
                               self_supervision=self_supervision,
                               output_transforms=output_transforms,
                               patch_size=patch_size,
                               input_size=input_size,
                               pin_memory=args.pin_mem,
                               drop_last=args.drop_last,
                               device=('cuda' if args.cuda else 'cpu'))

        return manager
