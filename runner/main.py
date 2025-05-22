from parcs.client import Task
import numpy as np
from PIL import Image

def main():
    # Читання та підготовка
    image = Image.open('cameraman.tif').convert('L')
    image_array = np.array(image) / 255.0
    h, w = image_array.shape

    block_size = 100
    blocks = []

    for i in range(0, h - block_size, block_size):
        block = image_array[i:i+block_size+2, :]  # +2 — padding
        blocks.append((i, block))

    task = Task()
    processes = []

    for i, block in blocks:
        proc = task.run_class("sobel_worker.SobelWorker")
        proc.send(block.tolist())
        processes.append((i, proc))

    # Очікуємо результати
    final_image = np.zeros_like(image_array)

    for i, proc in processes:
        result = np.array(proc.recv())
        final_image[i+1:i+result.shape[0]-1, :] = result[1:-1, :]

    # Порогове значення
    final_thresh = np.where(final_image > 0.5, 255, 0).astype(np.uint8)
    Image.fromarray(final_thresh).save("sobel_output.png")
    print("Готово: sobel_output.png")

if __name__ == "__main__":
    main()
