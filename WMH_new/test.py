from preprocess import preprocess_pm
from utils import * 

t1_cmap, fl_cmap = get_cmap()
preprocess_pm('C026C_B2_VNTR', t1_cmap, fl_cmap)
