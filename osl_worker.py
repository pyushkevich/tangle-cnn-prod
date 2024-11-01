#!/bin/python

import openslide
import queue
from PIL import Image
import SimpleITK as sitk

# Set up a queue for image chunks
chunk_q = queue.Queue(4)


# Define the worker function to load chunks from the openslide image
def osl_worker(osl, u_range, v_range, window_size_raw, padding_size_raw):
    for u in range(u_range[0], u_range[1]):
        for v in range(v_range[0], v_range[1]):
            # Get the coordinates of the window in raw pixels
            x, y, w = u * window_size_raw, v * window_size_raw, window_size_raw

            # Subtract the padding
            xp, yp, wp = x - padding_size_raw, y - padding_size_raw, window_size_raw + 2 * padding_size_raw

            # Read the chunk from the image
            if isinstance(osl, openslide.OpenSlide):
                chunk_img = osl.read_region((xp, yp), 0, (wp, wp)).convert("RGB")
            elif isinstance(osl, Image.Image):
                chunk_img = osl.crop((xp, yp, xp+wp, yp+wp))
            elif isinstance(osl, sitk.Image):
                print(osl.GetSize(), xp, yp, wp)
                crop = sitk.RegionOfInterest(osl, size=(wp,wp), index=(xp,yp))
                chunk_np = sitk.GetArrayFromImage(crop).astype('uint8')
                print(chunk_np.shape)
                chunk_img = Image.fromarray()
            else:
                raise ValueError('unknown type for osl in osl_worker')

            # Place into queue
            chunk_q.put(((u, v), (x, y, w), (xp, yp, wp), chunk_img))

    # Put a sentinel value
    chunk_q.put(None)


# Read from the OSL worker queue
def osl_read_chunk_from_queue():
    return chunk_q.get()


