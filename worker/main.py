from parcs.server import Service, serve
from PIL import Image
import numpy as np
import logging

class SobelWorker(Service):
    def sobel_filter(self, image):
        gray_image = image.convert("L")
        img_array = np.array(gray_image, dtype=np.float32)

        Kx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32)
        Ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float32)

        rows, cols = img_array.shape
        G = np.zeros((rows, cols), dtype=np.float32)

        for i in range(1, rows-1):
            for j in range(1, cols-1):
                region = img_array[i-1:i+2, j-1:j+2]
                gx = np.sum(Kx * region)
                gy = np.sum(Ky * region)
                G[i, j] = np.sqrt(gx**2 + gy**2)

        G = (G / G.max()) * 255
        G = G.astype(np.uint8)
        return Image.fromarray(G)

    def run(self):
        logging.info(f'Started recieving data')
        image_chunk, filt_x, filt_y, start_row, end_row = self.read_all()
        logging.info(f'Recived!')
        image_chunk = np.array(image_chunk)
        filt_x = np.array(filt_x)
        filt_y = np.array(filt_y)

        rows, cols = image_chunk.shape
        result_x = np.zeros_like(image_chunk)
        result_y = np.zeros_like(image_chunk)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                region = image_chunk[i-1:i+2, j-1:j+2]
                result_x[i, j] = np.sum(region * filt_x)
                result_y[i, j] = np.sum(region * filt_y)

        # Обрізаємо паддінг
        result_x = result_x[1:1 + (end_row - start_row)]
        result_y = result_y[1:1 + (end_row - start_row)]

        self.send((result_x.tolist(), result_y.tolist()))

if __name__ == "__main__":
    serve(SobelWorker())
