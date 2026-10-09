"""Build the python-version CIFAR folders that data/cifar.py reads from the
official *binary* CIFAR archives (e.g. a mirror when cs.toronto.edu is not
reachable).

The binary and python versions store the training images in the same order.
This script checks that against the CIFAR-N label files shipped in this repo
(clean labels recovered via image_order_c10/c100.npy) before writing anything.

Usage:
    python repro/prepare_cifar_from_binary.py --c10 DIR_WITH_data_batch_1.bin \
        --c100 DIR_WITH_train.bin --out ~/data
"""
import argparse
import os
import pickle

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_bin(path, label_bytes):
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, label_bytes + 3072)
    return raw[:, :label_bytes].astype(np.int64), np.ascontiguousarray(raw[:, label_bytes:])


def python_order_clean_labels(ordered_npy, order_npy):
    labels = np.load(ordered_npy, allow_pickle=True).item()['clean_label']
    order = np.load(order_npy)
    out = np.empty(len(order), dtype=np.int64)
    out[order] = np.asarray(labels).reshape(-1)
    return out


def dump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def build_c10(src, out):
    batches = [read_bin(os.path.join(src, f'data_batch_{i}.bin'), 1) for i in range(1, 6)]
    train_labels = np.concatenate([b[0][:, 0] for b in batches])
    expected = python_order_clean_labels(os.path.join(REPO, 'data/CIFAR-10_human_ordered.npy'),
                                         os.path.join(REPO, 'image_order_c10.npy'))
    assert np.array_equal(train_labels, expected), 'CIFAR-10 binary order does not match CIFAR-N labels'

    dst = os.path.join(out, 'cifar-10-batches-py')
    os.makedirs(dst, exist_ok=True)
    for i, (lab, data) in enumerate(batches, 1):
        dump({'data': data, 'labels': lab[:, 0].tolist()}, os.path.join(dst, f'data_batch_{i}'))
    lab, data = read_bin(os.path.join(src, 'test_batch.bin'), 1)
    dump({'data': data, 'labels': lab[:, 0].tolist()}, os.path.join(dst, 'test_batch'))
    with open(os.path.join(src, 'batches.meta.txt')) as f:
        names = [l.strip() for l in f if l.strip()]
    dump({'label_names': names, 'num_cases_per_batch': 10000, 'num_vis': 3072}, os.path.join(dst, 'batches.meta'))
    print(f'CIFAR-10 written to {dst} (50000/50000 train labels match CIFAR-N clean labels)')


def build_c100(src, out):
    lab, data = read_bin(os.path.join(src, 'train.bin'), 2)
    expected = python_order_clean_labels(os.path.join(REPO, 'data/CIFAR-100_human_ordered.npy'),
                                         os.path.join(REPO, 'image_order_c100.npy'))
    assert np.array_equal(lab[:, 1], expected), 'CIFAR-100 binary order does not match CIFAR-N labels'

    dst = os.path.join(out, 'cifar-100-python')
    os.makedirs(dst, exist_ok=True)
    dump({'data': data, 'fine_labels': lab[:, 1].tolist(), 'coarse_labels': lab[:, 0].tolist()},
         os.path.join(dst, 'train'))
    lab, data = read_bin(os.path.join(src, 'test.bin'), 2)
    dump({'data': data, 'fine_labels': lab[:, 1].tolist(), 'coarse_labels': lab[:, 0].tolist()},
         os.path.join(dst, 'test'))
    meta = {}
    for key, fname in [('fine_label_names', 'fine_label_names.txt'), ('coarse_label_names', 'coarse_label_names.txt')]:
        with open(os.path.join(src, fname)) as f:
            meta[key] = [l.strip() for l in f if l.strip()]
    dump(meta, os.path.join(dst, 'meta'))
    print(f'CIFAR-100 written to {dst} (50000/50000 train labels match CIFAR-N clean labels)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--c10', help='directory containing data_batch_{1..5}.bin and test_batch.bin')
    p.add_argument('--c100', help='directory containing train.bin and test.bin')
    p.add_argument('--out', default='~/data')
    args = p.parse_args()
    out = os.path.expanduser(args.out)
    if args.c10:
        build_c10(args.c10, out)
    if args.c100:
        build_c100(args.c100, out)
