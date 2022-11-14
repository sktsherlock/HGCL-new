import os.path as osp
import sys
import subprocess as sp
from pathlib import Path



PROJ_DIR = osp.abspath(osp.dirname(__file__))
print(PROJ_DIR)
PROJ_NAME = 'HGCL-new'
env_vars = {
    # PATHS
    'LP': PROJ_DIR,  # Local Path
    'PROJ_NAME': PROJ_NAME,  # PROJ Name
}

server_setting_file = f'{PROJ_DIR}/shell_env.sh'
with open(server_setting_file, 'w') as f:
    for var_name, var_val in env_vars.items():
        f.write(f'export {var_name}="{var_val}"\n')

    for cmd in SV_INIT_CMDS:
        f.write(f'{cmd}\n')
    print()
