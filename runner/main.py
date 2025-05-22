from parcs.server import Runner, serve

class SobelRunner(Runner):
    def run(self):
        image_path = self.args.get("img.png")
        if not image_path:
            return {"error": "No image path provided"}

        task = self.start("sirin027/sobel-worker:latest")
        task.send_all({"image": image_path})
        result = task.recv()
        task.shutdown()
        return result

if __name__ == "__main__":
    serve(SobelRunner)
