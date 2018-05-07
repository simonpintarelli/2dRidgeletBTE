#!/usr/bin/env python

import yaml
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generates config.yaml')
    parser.add_argument('-K', type=int, help="polynomial degree")

    args = parser.parse_args()
    K = args.K

    dirs = [[i, j] for i in range(0, K, 4) for j in range(0, K, 4)]

    data = {'dirs': dirs}

    qs = [1e-4, 1e-3, 1e-2]

    data['qs'] = qs
    f = open('config.yaml', 'w')

    print(yaml.dump(data), file=f)
