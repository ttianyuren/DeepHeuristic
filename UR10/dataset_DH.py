import os
import numpy as np
import torch
import torch.utils.data
import argparse
import h5py
import json
import sys
import pickle as pk


def rotate_point_cloud_so3(points):
    angles = np.random.uniform(0, 1, 3) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    points = np.dot(points.reshape((-1, 3)), R)
    return points


def rotate_perturbation_point_cloud(points, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
            Nx3 array, original point clouds
        Return:
            Nx3 array, rotated point clouds
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    points = np.dot(points.reshape((-1, 3)), R)
    return points


def rotate_point_cloud_z(points):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
            Nx3 array, original point clouds
        Return:
            Nx3 array, rotated point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])

    return np.dot(points.reshape((-1, 3)), rotation_matrix)


def getDataFiles(list_filename, root=''):
    file_names = [os.path.split(line.rstrip())[1] for line in open(list_filename)]
    if root:
        file_names = [os.path.join(root, fname) for fname in file_names]
    return file_names


def load_h5(h5_filename):
    f = h5py.File(h5_filename, "r")
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


def loadDataFileNames(filename, root=''):
    fpath = os.path.join(root, filename)
    with open(fpath) as f:
        names = json.load(f)
    return names


class Dataset_GraspPredict(torch.utils.data.Dataset):

    def __init__(self, file):
        with open(file, 'rb') as f:
            self.X, self.labels = pk.load(f)

    def __getitem__(self, index):
        """For each pointCloud instance, [ [n_point, n_dim], label ] -> [ [n_radius, n_vertex, n_feature], label ]"""

        points = self.X[index]
        label = self.labels[index]

        x = torch.FloatTensor(points)  # [n_vertex_l4=2562, n_feature=3 or 5]
        label = torch.FloatTensor(label).squeeze()  # [n_vertex_l1=42]

        items = [x, label]
        return items  #

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default=".", help="Target data directory")
    args = parser.parse_args()

    # Download ModelNet40 dataset for point cloud classification
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    if not os.path.exists(os.path.join(args.data_dir, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], args.data_dir))
