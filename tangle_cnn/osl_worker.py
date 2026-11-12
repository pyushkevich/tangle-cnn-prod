#!/bin/python

import openslide
import queue
from PIL import Image
import SimpleITK as sitk
import numpy as np
import parse

# Set up a queue for image chunks
chunk_q = queue.Queue(4)

class HistologyDataSource:
    
    def __init__(self):
        pass
    
    @property
    def dimensions(self):
        pass
    
    @property
    def spacing(self):
        pass
    
    def read_region(self, location, level, size):
        pass

    
class OpenSlideHistologyDataSource(HistologyDataSource):
    
    def __init__(self, osl):
        HistologyDataSource.__init__(self)
        self._osl = osl
        
    @property
    def dimensions(self):
        return np.array(self._osl.dimensions)
    
    @property
    def spacing(self):
        (sx, sy) = (0.0, 0.0)
        if 'openslide.mpp-x' in self._osl.properties:
            sx = float(self._osl.properties['openslide.mpp-x']) / 1000.0
            sy = float(self._osl.properties['openslide.mpp-y']) / 1000.0
        elif 'openslide.comment' in self._osl.properties:
            for z in self._osl.properties['openslide.comment'].split('\n'):
                r = parse.parse('Resolution = {} um', z)
                if r is not None:
                    sx = float(r[0]) / 1000.0
                    sy = float(r[0]) / 1000.0

        # If there is no spacing, throw exception
        if sx == 0.0 or sy == 0.0:
            raise Exception('No spacing information in image')
        
        return np.array((sx, sy))
    
    def read_region(self, location, level, size):
        return self._osl.read_region(location, level, size)


class SimpleITKHistologyDataSource(HistologyDataSource):
    
    def __init__(self, image):
        HistologyDataSource.__init__(self)
        self._src_image = image
        self._byte_image = Image.fromarray(sitk.GetArrayFromImage(image).astype('uint8'))
        
    @property
    def dimensions(self):
        return np.array(self._src_image.GetSize())[:2]
    
    @property
    def spacing(self):
        return np.array(self._src_image.GetSpacing())[:2]
    
    def read_region(self, location, level, size):
        x,y = location
        w,h = size
        return self._byte_image.crop((x,y,x+w,y+h))
        #crop = sitk.RegionOfInterest(self._byte_image, size=[w,h,1], index=[x,y,0])
        #chunk_np = sitk.GetArrayFromImage(crop).astype('uint8')
        #return Image.fromarray(chunk_np)

# Define the worker function to load chunks from the openslide image
def osl_worker(osl, u_range, v_range, window_size_raw, padding_size_raw, stop_event=None,
               transform=None, pin_memory=False):
    for u in range(u_range[0], u_range[1]):
        for v in range(v_range[0], v_range[1]):
            # Stop early if the consumer has given up (e.g., after an error)
            if stop_event is not None and stop_event.is_set():
                chunk_q.put(None)
                return

            # Get the coordinates of the window in raw pixels
            x, y, w = u * window_size_raw, v * window_size_raw, window_size_raw

            # Subtract the padding
            xp, yp, wp = x - padding_size_raw, y - padding_size_raw, window_size_raw + 2 * padding_size_raw

            # Read the chunk from the image
            chunk_img = osl.read_region((xp, yp), 0, (wp, wp)).convert("RGB")

            # Optionally convert the chunk to a network-ready tensor here, so that this
            # work overlaps with inference in the consumer thread
            if transform is not None:
                chunk_tensor = transform(chunk_img)
                if pin_memory:
                    chunk_tensor = chunk_tensor.pin_memory()
                chunk_q.put(((u, v), (x, y, w), (xp, yp, wp), chunk_img, chunk_tensor))
            else:
                chunk_q.put(((u, v), (x, y, w), (xp, yp, wp), chunk_img))

    # Put a sentinel value
    chunk_q.put(None)


# Read from the OSL worker queue
def osl_read_chunk_from_queue():
    return chunk_q.get()


# Stop the OSL worker and empty the queue, so that the worker thread can exit and
# the queue holds no stale chunks for the next run
def osl_stop_worker(worker, stop_event):
    stop_event.set()
    while worker.is_alive():
        try:
            chunk_q.get(timeout=0.1)
        except queue.Empty:
            pass
    while not chunk_q.empty():
        chunk_q.get_nowait()


