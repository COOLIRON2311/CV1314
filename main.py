import cv2
import numpy as np
import streamlit as st
from predictor import Predictor
from encoder import EncodingType


class NyandexImages:
    def __init__(self):
        self.predictor = Predictor(EncodingType.CLIP)
    
    def run(self):
        st.title('Nyandex Images')
        file = st.file_uploader('Drop an image here')
        if file is not None:
            buf = np.frombuffer(file.getbuffer(), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            st.image(img)
            st.title('10 most similar images')
            for similar_image in self.predictor.find_similar(img, 10):
                st.image(similar_image)


if __name__ == '__main__':
    nya = NyandexImages()
    nya.run()
