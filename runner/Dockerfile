FROM lionell/parcs-py

COPY test.tif .
COPY main.py .
RUN pip install Pillow numpy
CMD ["python", "main.py"]