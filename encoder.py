from enum import Enum
import joblib
import torch
import cv2
import numpy as np
import clip
from PIL import Image


class EncodingType(Enum):
    KMeans = 1
    CLIP = 2

class Coder:
    def __init__(self, encoding_type: EncodingType):
        self.encoding_type = encoding_type
        if self.encoding_type == EncodingType.KMeans:
            self.kmeans = joblib.load('kmeans_1000_512.joblib')
            self.sift = cv2.SIFT.create()
        elif encoding_type == EncodingType.CLIP:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
    def encode(self, image: np.ndarray) -> np.ndarray:
        if self.encoding_type == EncodingType.KMeans:
            return self._img_path_to_bovw(image)
        elif self.encoding_type == EncodingType.CLIP:
            return self._img_path_to_clip(image)
        
    def _img_path_to_bovw(self, image: np.ndarray) -> np.ndarray:
        _, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is None:
            default_hist = np.zeros(self.kmeans.n_clusters)
            default_hist[:] = 1 / self.kmeans.n_clusters
            return default_hist
        predictions = self.kmeans.predict(descriptors)
        histogram = np.bincount(predictions, minlength=self.kmeans.n_clusters)
        histogram = histogram / np.sum(histogram)
        return histogram
    
    def _img_path_to_clip(self, image: np.ndarray) -> np.ndarray:
        image = self.preprocess(Image.fromarray(np.uint8(image))).unsqueeze(0).to(self.device)
        return self.model.encode_image(image).cpu().detach().numpy().squeeze().astype(np.float64)
    