import numpy as np
from fastai.dataloader import DataLoader


class DialogueDataLoader(DataLoader):

    def pad2d(self, x, max_cl, max_sl):

        paddings = [(max_cl - x.shape[0], 0), (max_sl - x.shape[1], 0)] if self.pre_pad else [(0, max_cl - x.shape[0]),
                                                                                              (0, max_sl - x.shape[1])]
        return np.pad(x, paddings, mode='constant', constant_values=self.pad_idx)

    def pad1d(self, x, max_sl):
        paddings = [(max_sl - x.size, 0)] if self.pre_pad else [(0, max_sl - x.size)]
        return np.pad(x, paddings, mode='constant', constant_values=self.pad_idx)

    def get_batch(self, indices):
        max_cl, max_sl = np.asarray([self.dataset[i][0].shape for i in indices]).max(axis=0)
        x_batch = np.stack([self.pad2d(self.dataset[i][0], max_cl, max_sl) for i in indices], axis=0)
        max_sl = max([self.dataset[i][1].size for i in indices])
        y_batch = np.stack([self.pad1d(self.dataset[i][1], max_sl) for i in indices], axis=0)
        y_target = np.stack([self.pad1d(self.dataset[i][2], max_sl) for i in indices], axis=0)
        res = [x_batch, y_batch, y_target]
        if self.transpose:
            res[0], res[1] = np.transpose(res[0], [1, 2, 0]), res[1].T
        if self.transpose_y:
            res[2] = res[2].T
        return res
