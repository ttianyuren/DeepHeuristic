import torch
import torch.nn.functional as F
from torch import nn
import argparse
import os
import logging
import sys
import time
import numpy as np
from model_DH import SGNN
from mesh.mesh import icosphere
from dataset_DH import Dataset_GraspPredict

import gc

N_CLASS = 40


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_step(model, data_b, target_b, optimizer):
    model.train()
    data_b, target_b = data_b.cuda(), target_b.cuda()

    output_b = torch.squeeze(model(data_b))

    # prediction_b = F.log_softmax(output_b, dim=-1)
    prediction_b = F.sigmoid(output_b)

    # loss = F.nll_loss(prediction_b, target_b)  # TODO: change the loss function!
    loss = F.binary_cross_entropy(prediction_b, target_b)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    correct = prediction_b.data.round().eq(target_b.data).long().cpu().sum()

    return loss.item(), correct.item()


# @torch.inference_mode()
def test_step(model, data_b, target_b):
    model.eval()

    data_b, target_b = data_b.cuda(), target_b.cuda()

    with torch.no_grad():
        output_b = torch.squeeze(model(data_b))

        prediction_b = F.sigmoid(output_b)
        loss = F.binary_cross_entropy(prediction_b, target_b)

        correct = prediction_b.data.round().eq(target_b.data).long().cpu().sum()

    return loss, correct


def get_log_name(args):
    out_dir = os.path.join(args.data_dir, "GraspPredict_out")
    log_dir = os.path.join(out_dir, "logs")
    if args.out:
        log_dir = os.path.join(log_dir, args.out)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    lr = args.lr
    dropout = args.dropout
    weight_decay = args.weight_decay
    param_factor = args.F

    log_name = "l{}-F{}-b{}-lr{:.2e}-{}".format(args.l, param_factor,
                                                args.batch_size, lr, args.rotate_train)

    if args.p != "max":
        log_name += "-p_{}".format(args.p)
    if dropout:
        log_name += "-drop{:.2f}".format(dropout)
    if weight_decay:
        log_name += "-wd{:.2e}".format(weight_decay)
    if args.seed:
        log_name += "-s{}".format(args.seed)
    if args.identifier:
        log_name += "-{}".format(args.identifier)

    return log_dir, log_name


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)


def model_dims(n_levels, base_dim, factor=1):
    dims_init = [base_dim * (2 ** i) for i in range(n_levels + 1)]
    dims = [factor * dim for dim in dims_init]
    return dims


def model_setup(args, **kwargs):
    dropout = args.dropout
    param_factor = args.F
    n_hidden = 1
    gconv = args.g
    pool_type = args.p

    mesh = icosphere(args.l)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    BASE_DIM = 8
    conv_block_dims = model_dims(args.l, BASE_DIM, param_factor)  # [32, 64, 128, 256, 512] for level=4
    conv_depth_dims = [1, 1, 1, 1, 0]

    output_hidden_dims = []
    if n_hidden == 1:
        output_hidden_dims = [512]
    if n_hidden == 2:
        output_hidden_dims = [512, 256]

    out_dim = 1
    in_feat_dim = 5

    model = SGNN(mesh, 1, gconv, conv_block_dims, conv_depth_dims, output_hidden_dims,
                 in_feat_dim, out_dim, pool_type, dropout)

    # model.cpu()
    model.cuda()

    return model


def prepare_data(args):
    train_data = Dataset_GraspPredict('train_Xs_labels.pk')
    valid_data = Dataset_GraspPredict('test_Xs_labels.pk')

    num_workers = args.workers
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
                                               worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size,
                                               num_workers=num_workers, pin_memory=True, drop_last=False,
                                               worker_init_fn=worker_init_fn)

    n_train, n_valid = len(train_data), len(valid_data)

    return train_loader, valid_loader


def train(args):
    dropout = args.dropout
    lr = args.lr
    wd = args.weight_decay
    param_factor = args.F

    train_loader, valid_loader = prepare_data(args)
    model = model_setup(args)

    """ logging """
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console = logging.StreamHandler(sys.stdout)
    logger.addHandler(console)
    log_dir, log_name = get_log_name(args)
    save_model_name = log_name + ".pkl"
    log_fname = "{}.txt".format(log_name)
    log_path = os.path.join(log_dir, log_fname)

    fh = logging.FileHandler(os.path.join(log_dir, log_fname))
    logger.addHandler(fh)

    logger.info("Parameters: {}".format(count_parameters(model)))
    logger.info("{}\n".format(model))

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    DECAY_FACTOR = 0.1
    stop_thresh = 1e-3
    decay_patience = 4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           factor=DECAY_FACTOR,
                                                           patience=decay_patience, threshold=stop_thresh)

    best_val_loss = 1e10
    best_val_acc = -1
    for epoch in range(args.epochs):
        total_loss = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        n_batches = 0
        n_seen = 0

        time_before_train = time.perf_counter()

        for batch_idx, (data, target) in enumerate(train_loader):
            time_after_load = time.perf_counter()
            n_batches += 1
            total_label = target.shape[0] * target.shape[1]
            n_seen += total_label
            time_before_step = time.perf_counter()
            loss, correct = train_step(model, data, target, optimizer)
            total_loss += loss
            total_correct += correct

            logger.info("[{}:{}/{}] LOSS={:.3} <LOSS>={:.3} ACC={:.3} <ACC>={:.3} time={:.2}+{:.2}".format(
                epoch, batch_idx, len(train_loader),
                loss, total_loss / n_batches,
                      correct / total_label, total_correct / n_seen,
                      time_after_load - time_before_load,
                      time.perf_counter() - time_before_step))

            time_before_load = time.perf_counter()
            if args.debug:
                break

        logger.info("mode=training epoch={} lr={:.2e} <LOSS>={:.3} <ACC>={:.3f} time={:.3f}".format(
            epoch,
            lr,
            total_loss / n_batches,
            total_correct / n_seen,
            time.perf_counter() - time_before_train))

        # test
        total_loss = 0
        total_correct = 0
        n_batches = 0
        n_seen = 0

        time_before_eval = time.perf_counter()
        for batch_idx, (data, target) in enumerate(valid_loader):
            loss, correct = test_step(model, data, target)
            total_loss += loss
            n_correct = correct.sum().item()
            total_correct += n_correct
            n_batches += 1
            total_label = target.shape[0] * target.shape[1]
            n_seen += total_label

            if args.debug:
                break

        total_loss_avg = total_loss / n_batches
        if total_loss_avg < best_val_loss:
            best_val_loss = total_loss_avg

        val_acc = total_correct / n_seen
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), os.path.join(log_dir, save_model_name))
            best_val_acc = val_acc

        logger.info(
            "mode=validation epoch={} lr={:.3} <LOSS>={:.3} *LOSS={:.3} <ACC>={:.3f} *ACC={:.3f} time={:.3}".format(
                epoch,
                lr,
                total_loss_avg,
                best_val_loss,
                val_acc,
                best_val_acc,
                time.perf_counter() - time_before_eval))

        lr_curr = optimizer.param_groups[0]['lr']
        if lr_curr < 1e-5:
            logger.info("Early termination at LR={}".format(lr_curr))
            break

        scheduler.step(total_loss)

    logger.info("Final: *ACC={:3f}".format(best_val_acc))

    return best_val_loss


if __name__ == "__main__":
    # print(torch.version.cuda)

    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default=".", help="Data directory")
    parser.add_argument("-o", "--out", help="Name of output directory")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)  # 32
    parser.add_argument('-l', '--level', type=int, dest='l', default=4, help='Level of mesh refinement')
    parser.add_argument("-p", default="max", choices=["max", "mean", "sum"], help="Pooling type: [max, sum, mean]")
    parser.add_argument("-F", type=int, help="Controls model layer parameters", default=4)
    parser.add_argument("-g", help="Graph convolution type", choices=["gcn", "graphsage"], default="gcn")
    parser.add_argument('-e', '--R_expand', help='Radial expansion factor for kernel', type=int, default=8)
    parser.add_argument('-T', '--kernel-thresh', help='Threshold for kernel mapping', type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=2.2e-4)
    parser.add_argument("--dropout", type=float, default=0.14)
    parser.add_argument("--weight-decay", type=float, default=2.7e-7)
    parser.add_argument("--seed", type=int, help='random seed', default=24)
    parser.add_argument("--num-votes", type=int, help='Number of votes during testing (rotation only)', default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rotate_train", default="SO3", choices=["NR", "z", "SO3"])
    parser.add_argument("--rotate_test", default="SO3", choices=["NR", "z", "SO3"])
    parser.add_argument('--device', default=0, type=int, help='GPU device ID')
    parser.add_argument("--workers", type=int, default=6, help="Number of processes for data loading")
    parser.add_argument("--debug", action='store_true', help="Debug mode")
    parser.add_argument("--identifier", help="user-defined string to add to log name")
    args = parser.parse_args()

    train(args)
