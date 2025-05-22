from parcs.server import Service, serve
from PIL import Image
import numpy as np

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
        image_path = self.args.get("image")
        if not image_path:
            return {"error": "No image path provided"}

        try:
            image = Image.open(image_path)
            processed_image = self.sobel_filter(image)
            output_path = f"processed_{image_path}"
            processed_image.save(output_path)
            return {"status": "success", "output": output_path}
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    serve(SobelWorker)
