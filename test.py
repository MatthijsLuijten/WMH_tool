import WMH_new
from WMH_new import parameters

import os

for i in range(3):
    print(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(i+1)).replace("\\","/"))
