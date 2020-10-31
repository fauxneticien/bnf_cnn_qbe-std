import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist

class STD_Dataset(Dataset):
    """Spoken Term Detection dataset."""

    def __init__(self, root_dir, labels_csv, query_dir, audio_dir, apply_vad = False, max_height = 100, max_width = 800):
        """
        Args:
            root_dir (string): Absolute path to dataset directory with content below
            labels_csv (string): Relative path to the csv file with query and test pairs, and labels
                (1 = query in test; 0 = query not in test).
            query_dir (string): Relative path to directory with all the audio queries.
            audio_dir (string): Relative path to directory with all the test audio.
        """
        self.qtl_frame  = pd.read_csv(os.path.join(root_dir, labels_csv))
        self.query_dir  = os.path.join(root_dir, query_dir)
        self.audio_dir  = os.path.join(root_dir, audio_dir)
        self.apply_vad  = apply_vad
        self.max_height = max_height
        self.max_width  = max_width

        if apply_vad is True:
            # If using voice activity detection we expect same directory structure
            # and file names as feature files for .npy files containing voice activity
            # detection (VAD) labels (0 = no speech activity, 1 = speech activity)
            # in a 'vad_labels' directory
            self.vad_query_dir = os.path.join(root_dir, 'vad_labels', query_dir)
            self.vad_audio_dir = os.path.join(root_dir, 'vad_labels', audio_dir)

    def __len__(self):
        return len(self.qtl_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        query_name = self.qtl_frame.iloc[idx, 0] 
        test_name  = self.qtl_frame.iloc[idx, 1] 
        qt_label   = self.qtl_frame.iloc[idx, 2]

        # Get features where query = M x f, test = N x f, where M, N number of frames and f number of features
        query_feats = np.load(os.path.join(self.query_dir, query_name + ".npy"), allow_pickle=True)
        test_feats  = np.load(os.path.join(self.audio_dir, test_name + ".npy"), allow_pickle=True)

        if self.apply_vad is True:
            query_vads = np.load(os.path.join(self.vad_query_dir, query_name + ".npy"), allow_pickle=True)
            test_vads  = np.load(os.path.join(self.vad_audio_dir, test_name + ".npy"), allow_pickle=True)

            # Keep only frames (rows, axis = 0) where voice activity detection by rVAD has returned non-zero (i.e. 1)
            query_feats = np.take(query_feats, np.flatnonzero(query_vads), axis = 0)
            test_feats  = np.take(test_feats, np.flatnonzero(test_vads), axis = 0)

        # Create standardised Euclidean distance matrix of dimensions M x N
        qt_dists    = cdist(query_feats, test_feats, 'seuclidean', V = None)
        # Range normalise matrix to [-1, 1]
        qt_dists    = -1 + 2 * ((qt_dists - qt_dists.min())/(qt_dists.max() - qt_dists.min()))

        # Get indices to downsample or pad M x N matrix to max_height x max_width (default 100 x 800)
        def get_keep_indices(dim_size, dim_max):
            if dim_size <= dim_max:
                # no need to downsample if M or N smaller than max_height/max_width
                return np.arange(0, dim_size)
            else:
                # if bigger, return evenly spaced indices for correct height/width
                return np.round(np.linspace(0, dim_size - 1, dim_max)).astype(int)

        ind_rows = get_keep_indices(qt_dists.shape[0], self.max_height)
        ind_cols = get_keep_indices(qt_dists.shape[1], self.max_width)

        qt_dists = np.take(qt_dists, ind_rows, axis = 0)
        qt_dists = np.take(qt_dists, ind_cols, axis = 1)

        # Create empty 100 x 800 matrix, then fill relevant cells with dist values
        temp_dists = np.full((self.max_height, self.max_width), qt_dists.min(), dtype='float32')
        temp_dists[:qt_dists.shape[0], :qt_dists.shape[1]] = qt_dists

        # Reshape to (1xHxW) since to feed into ConvNet with 1 input channel
        dists = torch.Tensor(temp_dists).view(1, self.max_height, self.max_width)
        label = torch.Tensor([qt_label])

        sample = {'query': query_name, 'reference': test_name, 'dists': dists, 'labels': label}

        return sample
