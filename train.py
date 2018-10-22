
import argparse
import time
import matplotlib.pyplot as plt
import logging
import imp
import dataset
import utils
import proxynca
import net

import torch
import torch.utils.data
import numpy as np
import matplotlib
matplotlib.use('agg', warn=False, force=True)

parser = argparse.ArgumentParser(description='Training inception V2' +
    'as described in `No Fuss Distance Metric Learning using Proxies.`'
)
# export directory, training and val datasets, test datasets
parser.add_argument('--dataset', 
    default='UPMC-G20',
    help='Name of the dataset (UPMC-G20 or CUB200)'
)
parser.add_argument('--root-folder',
    default='image_data',
    help='Path to the dataset folder, containing the images folder.'
)
parser.add_argument('--is-extracted', action='store_true',
    default=False,
    help='If image data was already extracted, do not extract it again.' +
    ' Otherwise use extracted data.'
)
parser.add_argument('--embedding-size', default=64, type=int,
    dest='sz_embedding',
    help='Size of embedding that is appended to InceptionV2.'
)
parser.add_argument('--number-classes', default=10, type=int,
    dest='nb_classes',
    help='Number of first [0, N] classes used for training and ' +
    'next [N, N * 2] classes used for evaluating.'
)
parser.add_argument('--batch-size', default=32, type=int,
    dest='sz_batch',
    help='Number of samples per batch.'
)
parser.add_argument('--lr-embedding', default=1e-5, type=float,
    help='Learning rate for embedding.'
)
parser.add_argument('--lr-inception', default=1e-3, type=float,
    help='Learning rate for Inception, excluding embedding layer.'
)
parser.add_argument('--lr-proxynca', default=1e-3, type=float,
    help='Learning rate for proxies of Proxy NCA.'
)
parser.add_argument('--weight-decay', default=5e-4, type=float,
    dest='weight_decay',
    help='Weight decay for Inception, embedding layer and Proxy NCA.'
)
parser.add_argument('--epsilon', default=1e-2, type=float,
    help='Epsilon (optimizer) for Inception, embedding layer and Proxy NCA.'
)
parser.add_argument('--gamma', default=1e-1, type=float,
    help='Gamma for multi-step learning-rate-scheduler.'
)
parser.add_argument('--epochs', default=20, type=int,
    dest='nb_epochs',
    help='Number of training epochs.'
)
parser.add_argument('--log-filename', default='example',
    help='Name of log file.'
)
parser.add_argument('--gpu-id', default=0, type=int,
    help='ID of GPU that is used for training.'
)
parser.add_argument('--workers', default=16, type=int,
    dest='nb_workers',
    help='Number of workers for dataloader.'
)

args = parser.parse_args()
torch.cuda.set_device(args.gpu_id)

imp.reload(logging)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
        logging.StreamHandler()
    ]
)

if args.dataset == 'UPMC-G20':
    type = 'Food'
elif args.dataset == 'CUB200':
    type = 'Birds'
else:
    logging.error('Unknown dataset. Use "UPMC-G20" or "CUB200" instead.')
    exit()

dl_tr = torch.utils.data.DataLoader(
    dataset.Data.factory(
        type=type,
        root=args.root_folder,
        labels=list(range(0, args.nb_classes)),
        is_extracted=args.is_extracted,
        transform=dataset.utils.make_transform()
    ),
    batch_size=args.sz_batch,
    shuffle=True,
    num_workers=args.nb_workers,
    drop_last=True,
    pin_memory=True
)

dl_ev = torch.utils.data.DataLoader(
    dataset.Data.factory(
        type=type,
        root=args.root_folder,
        labels=list(range(args.nb_classes, 2 * args.nb_classes)),
        is_extracted=args.is_extracted,
        transform=dataset.utils.make_transform(
        is_train=False)
    ),
    batch_size=args.sz_batch,
    shuffle=False,
    num_workers=args.nb_workers,
    pin_memory=True
)

model = net.bn_inception(pretrained=True)
net.embed(model, sz_embedding=args.sz_embedding)
model = model.cuda()

criterion = proxynca.ProxyNCA(args.sz_embedding, args.nb_classes,
                              args.sz_batch).cuda()

opt = torch.optim.Adam(
    [
        {  # embedding parameters
            'params': model.embedding_layer.parameters(),
            'lr': args.lr_embedding
        },
        {  # proxy nca parameters
            'params': criterion.parameters(),
            'lr': args.lr_proxynca
        },
        {  # inception parameters, excluding embedding layer
            'params': list(
                set(
                    model.parameters()
                ).difference(
                    set(model.embedding_layer.parameters())
                )
            ),
            'lr': args.lr_inception
        }
    ],
    eps=args.epsilon,
    weight_decay=args.weight_decay
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [3, 10, 16],
                                                 gamma=args.gamma)

logging.info("Training parameters: {}".format(vars(args)))
logging.info("Training for {} epochs.".format(args.nb_epochs))
losses = []
scores = []
scores_tr = []

t1 = time.time()
logging.info("**Evaluating initial model...**")
with torch.no_grad():
    utils.evaluate(model, dl_ev, args.nb_classes)

for e in range(1, args.nb_epochs + 1):
    scheduler.step()
    time_per_epoch_1 = time.time()
    losses_per_epoch = []
    for x, y in dl_tr:
        opt.zero_grad()
        m = model(x.cuda())
        loss = criterion(m, y.cuda())
        loss.backward()
        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()
    time_per_epoch_2 = time.time()
    losses.append(np.mean(losses_per_epoch[-20:]))
    logging.info(
        "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
            e,
            losses[-1],
            time_per_epoch_2 - time_per_epoch_1
        )
    )
    with torch.no_grad():
        logging.info("**Evaluating...**")
        scores.append(utils.evaluate(model, dl_ev, args.nb_classes))
    model.losses = losses
    model.current_epoch = e

t2 = time.time()
logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
