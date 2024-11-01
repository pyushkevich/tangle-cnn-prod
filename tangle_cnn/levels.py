import openslide as os
import sys

img=os.OpenSlide(sys.argv[1])
print("%s %d %d %d" % (sys.argv[1], img.dimensions[0], img.dimensions[1], img.level_count))
