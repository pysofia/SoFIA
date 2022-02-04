import os, shutil
import subprocess
from subprocess import call

class cd:
    """Context manager for changing the current working directory""" 
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

root = '/Users/anabel/Documents/PhD/Code'
dirName = root + '/SoFIA/installation/build/lib/sofia'
shutil.copy(root + '/SoFIA/Iprop/sampler.py',dirName)
shutil.copy(root + '/SoFIA/Probability_distributions/distributions.py',dirName)
shutil.copy(root + '/SoFIA/SA/Sobol.py',dirName)
shutil.copy(root + '/SoFIA/Fprop/pc.py',dirName)
shutil.copy(root + '/SoFIA/Applications/nitridation/models.py',dirName)
shutil.copy(root + '/SoFIA/Applications/nitridation/data_assembly.py',dirName)

# ---------------------------------
# Update SoFIA

with cd(root + "/SoFIA/installation"):

    cmd = ['python3', 'setup.py', 'install']

    subprocess.call(cmd)