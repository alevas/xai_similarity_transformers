import os
import logging
import torch
import datetime
import matplotlib

# Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# output_base_dir = os.path.join(ROOT_DIR, "outputs")
# instruction_path = "instructions"

# logging stuff
logging_datetime_format = '%Y%m%d__%H%M%S'
logging_time_format = '%H:%M:%S'
exp_start_time = datetime.datetime.now().strftime(logging_datetime_format)
log_level = logging.INFO
log_format = '%(asctime)10s,%(msecs)-3d %(module)-30s %(levelname)s %(message)s'
# log_model_limit = 2  # best and last epoch

# torch stuff
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Current variables for universal access
# current_exp_name = ""
# current_out_dir = ""  # set at the beggining of an experiment

# # plotting stuff
# cmap = matplotlib.cm.get_cmap('coolwarm')
# coolwarm_blue = [cmap(0)]
# coolwarm_red = [cmap(0.99)]
