from parcs.server import Service, serve
import numpy as np

class SobelWorker(Service):
    def apply_filter(self, image, filt):
        filtered_image = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                filtered_image[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * filt)
        return filtered_image

    def run(self):
        # Приймаємо блок зображення
        image_block = np.array(self.recv())

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        gx = self.apply_filter(image_block, sobel_x)
        gy = self.apply_filter(image_block, sobel_y)
        grad = np.sqrt(gx ** 2 + gy ** 2)

        # Повертаємо результат
        self.send(grad.tolist())

serve(SobelWorker())
