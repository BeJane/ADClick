import math

from kmeans_pytorch import kmeans
from model import backbones
from model.common import NetworkFeatureAggregator
import torch

def prototype_init(device):


    num_clusters=math.ceil(0.1*images.shape[0])

    features = [features[layer] for layer in layers_to_extract_from]
    cluster_ids_x, cluster_centers = kmeans(
        X=features[0], num_clusters=num_clusters, distance='l2', device=device,iter_limit=300,tol=1e-10
    )

