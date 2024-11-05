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
        crop = sitk.RegionOfInterest(self._byte_image, size=(wp,wp), index=(xp,yp))
        chunk_np = sitk.GetArrayFromImage(crop).astype('uint8')
        return Image.fromarray(chunk_np)

# Define the worker function to load chunks from the openslide image
def osl_worker(osl, u_range, v_range, window_size_raw, padding_size_raw):
    for u in range(u_range[0], u_range[1]):
        for v in range(v_range[0], v_range[1]):
            # Get the coordinates of the window in raw pixels
            x, y, w = u * window_size_raw, v * window_size_raw, window_size_raw

            # Subtract the padding
            xp, yp, wp = x - padding_size_raw, y - padding_size_raw, window_size_raw + 2 * padding_size_raw

            # Read the chunk from the image
            chunk_img = osl.read_region((xp, yp), 0, (wp, wp)).convert("RGB")

            # Place into queue
            chunk_q.put(((u, v), (x, y, w), (xp, yp, wp), chunk_img))

    # Put a sentinel value
    chunk_q.put(None)


# Read from the OSL worker queue
def osl_read_chunk_from_queue():
    return chunk_q.get()


