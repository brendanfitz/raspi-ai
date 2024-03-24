from pathlib import Path
import shutil

nas_wd = Path.home() / 'nas'
filename = 'test.h264'

shutil.copy(filename, nas_wd / filename)