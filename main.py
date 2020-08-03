import argparse
from json_config import CONFIG
from pynet.metrics import METRICS
from training import BaseTrainer
from testing import BaseTester, AlphaWGANTester, RobustnessTester, BayesianTester, EnsemblingTester
import torch
import logging

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default=CONFIG['input_path'])
    parser.add_argument("--metadata_path", type=str, default=CONFIG['metadata_path'])
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--freeze_until_layer", type=str)
    parser.add_argument("--checkpoint_dir", type=str, default="/neurospin/psy_sbox/bd261576/checkpoints")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--outfile_name", type=str, help="The output file name used to save the results in testing mode.")
    parser.add_argument("--N_train_max", type=int, default=None, help="Maximum number of training samples "
                                                                      "to be used per fold")
    parser.add_argument("--nb_epochs_per_saving", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--da", type=str, nargs='+', default=[], choices=['flip'])
    parser.add_argument("--manual_seed", type=int, help="The manual seed to give to pytorch.")
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate to use thoughout the network. "
                                                                 "Careful, all networks do not have this option.")
    parser.add_argument('--concrete_dropout', action='store_true')
    parser.add_argument("--bayesian", action='store_true', help="Whether to use dropout during test time or not")
    parser.add_argument("--nb_folds", type=int, default=5)
    parser.add_argument("--gamma_scheduler", type=float, required=True)
    parser.add_argument("--step_size_scheduler", type=int, default=10)
    parser.add_argument("--pin_mem", type=bool, default=True)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--db", choices=list(CONFIG['db'].keys()), required=True)
    parser.add_argument("--sampler", choices=["random", "weighted_random"], required=True)
    parser.add_argument("--model", choices=['base', 'SimCLR'], default='base')
    parser.add_argument("--labels", nargs='+', type=str, help="Label(s) to be predicted")
    parser.add_argument("--loss", type=str, choices=['BCE', 'l1', 'BCE_concrete_dropout', 'NTXenLoss', 'multi_l1_bce',
                                                     'l1_sup_NTXenLoss'], required=True)
    parser.add_argument("--folds", nargs='+', type=int, help="Fold indexes to run during the training")
    parser.add_argument("--stratify_label", type=str, help="Label used for the stratification of the train/val split")
    parser.add_argument("--with_visualization", action="store_true")
    parser.add_argument("--metrics", nargs='+', type=str, choices=list(METRICS.keys()), help="Metrics to be computed at each epoch")
    parser.add_argument("--add_input", action="store_true", help="Whether to add the input data to the output "
                                                                 "(e.g for a reconstruction task)")
    parser.add_argument("--num_cpu_workers", type=int, default=0, help="Number of workers assigned to do the "
                                                                       "preprocessing step (used by DataLoader of Pytorch)")
    parser.add_argument("--test_all_epochs", action="store_true")
    parser.add_argument("--test_best_epoch", type=str, choices=list(METRICS.keys()),
                        help="If set, it must be a metric or 'loss' in order to select the best epoch to test")
    parser.add_argument("--net", type=str, help="Initial learning rate")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--nb_epochs", type=int)
    parser.add_argument("--load_optimizer", action="store_true", help="If <pretrained_path> is set, loads also the "
                                                                      "optimizer's weigth")
    parser.add_argument("--with_logit", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--cuda", type=bool, default=True)

    # Kind of tests
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", choices=['basic', 'robustness', 'ssim', 'MC', 'ensemble'],
                        help="What kind of test it will perform.")

    args = parser.parse_args()
    logger = logging.getLogger("pynet")

    if args.test_best_epoch is not None:
        assert args.test_best_epoch in (args.metrics or []), \
            "--test_best_epoch must be chosen in {}".format((args.metrics or []))
        print("!!WARNING: For {}, it is assumed that the highest score is the best !!".format(args.test_best_epoch),
              flush=True)

    if args.manual_seed:
        torch.manual_seed(args.manual_seed)

    if not args.train and not args.test:
        args.train = True
        logger.info("No mode specify: training mode is set automatically")

    if args.train:
        trainer = BaseTrainer(args)
        trainer.run()
        # do not consider the pretrained path anymore since it will be eventually computed automatically
        args.pretrained_path = None

    if args.test == 'basic':
        tester = BaseTester(args)
        tester.run()

    if args.test == 'MC':
        tester = BayesianTester(args)
        tester.run()

    if args.test == "ensemble":
        tester = EnsemblingTester(args)
        tester.run()

    if args.test == 'ssim':
        tester = AlphaWGANTester(args)
        tester.run()

    if args.test == 'robustness':
        tester = RobustnessTester(args)
        tester.run()






