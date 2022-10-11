import WMH_new
from WMH_new import parameters

import os

print(os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/"))
