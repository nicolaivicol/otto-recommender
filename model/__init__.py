import os
from utils import set_display_options


set_display_options()

# allow using all threads in numexpr (used by word2vec / annoy)
if 'NUMEXPR_MAX_THREADS' not in os.environ.keys():
    import multiprocessing
    os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())
