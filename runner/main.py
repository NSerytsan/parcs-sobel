from parcs.server import Runner, serve
import os
import logging
import time
from PIL import Image
import numpy as np




def load_image(file_path):
    with Image.open(file_path) as image:
        image.load()
    return image

def split_image(image, num_parts):
    rows = image.shape[0]
    chunk_size = rows // num_parts
    chunks = []

    for i in range(num_parts):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_parts - 1 else rows

        padded_start = max(0, start - 1)
        padded_end = min(rows, end + 1)
        chunk = image[padded_start:padded_end]
        chunks.append((chunk, start, end))

    return chunks

class SobelRunner(Runner):

    def run(self):
        #
        image_file = os.environ.get('IMAGE_FILE', '/test.tif')
        p = int(os.environ.get('P', 3))
        logging.info(f'Loading image file {image_file}')
        #image = load_image(image_file)
        image = Image.open(image_file)
        image_gray = image.convert('L')
        image_array = np.array(image_gray) / 255.0
        chunks = split_image(image_array, p)
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        logging.info(f'Computing Sobel for {p} parts')

        #parts = split_sources(sources, p)
        #

        start_time = time.time()
        tasks = []
        for chunk_data in chunks:
            logging.info(f'Started sending tasks')
            chunk, start_row, end_row = chunk_data
            t = self.engine.run("sirin027/sobel-worker:latest")
            logging.info(f'Started worker')
            t.send_all(chunk.tolist(), sobel_x, sobel_y, start_row, end_row)
            logging.info(f'Started sending tasks')
            tasks.append(t)
        #task = self.engine.run("sirin027/sobel-worker:latest")
        #task.send_all({"image": image_path})
        #result = task.recv()
        #task.shutdown()
        sobel_x_rows = []
        sobel_y_rows = []
        logging.info(f"Tasks sent successfully")
        for t in tasks:
            sobel_x_part, sobel_y_part = t.recv()
            sobel_x_rows.extend(sobel_x_part)
            sobel_y_rows.extend(sobel_y_part)
            t.shutdown()
        sobel_x_result = np.array(sobel_x_rows)
        sobel_y_result = np.array(sobel_y_rows)
        sobel_combined = np.sqrt(sobel_x_result ** 2 + sobel_y_result ** 2)
        sobel_combined = (255 * sobel_combined / np.max(sobel_combined)).astype(np.uint8)
        final_image = Image.fromarray(sobel_combined)
        final_image.save("sobel_combined.png")
        logging.info(f'Saved filtered image to sobel_combined.png')
        logging.info(f"end time: {time.time() - start_time}")

        #return result


serve(SobelRunner())
