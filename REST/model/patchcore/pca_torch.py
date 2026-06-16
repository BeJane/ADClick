import numpy as np
import torch
from sklearn.decomposition import PCA


class PCA_torch():
    def __init__(self, n_components,device='cuda'):
        self.pca = PCA(n_components=n_components)
        self.device = device
    def fit_transform(self,features):
        new_features = self.pca.fit_transform(features)
        self.mean_ = torch.from_numpy(self.pca.mean_).to(self.device)
        self.components_ = torch.from_numpy(self.pca.components_).to(self.device)
        self.explained_variance_ = torch.from_numpy(self.pca.explained_variance_).to(self.device)
        return new_features

    def transform(self,X):
        if self.mean_ is not None:
            X = X - self.mean_

        X_transformed = torch.matmul(X, self.components_.T)

        if self.pca.whiten:
            X_transformed /= torch.sqrt(self.explained_variance_)

        return X_transformed