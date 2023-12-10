from utils import *
from encoder import EncodingType, Coder
from sklearn.neighbors import NearestNeighbors
import threading


class Predictor:
    def __init__(self, encoding_type: EncodingType):
        self.encoding_type = encoding_type
        self._set_params()
        # self.initialization = threading.Thread(target=self._set_params, name="Init", args=())
        # self.initialization.start()
        
    def _set_params(self):
        self.coder = Coder(self.encoding_type)
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
        if self.encoding_type == EncodingType.KMeans:
            self.df = get_from_csv('db_kmeans.csv')
        elif self.encoding_type == EncodingType.CLIP:
            self.df = get_from_csv('db_clip_big.csv')
        self.nn_model.fit(np.vstack(self.df['vector'].values), self.df.index.values)
        
    def find_similar(self, image: np.ndarray, similar_number: int = 10):
        # while(self.initialization.is_alive()):
        #     pass
        encoded_image = self.coder.encode(image)
        _, indices = self.nn_model.kneighbors([encoded_image], n_neighbors=similar_number)
        for idx in indices.flat:
            yield load_rbg(self.df.loc[idx]['path'])
