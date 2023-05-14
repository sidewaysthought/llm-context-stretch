from setuptools import setup
from setuptools.command.install import install
import subprocess

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        subprocess.call(['python', '-m', 'spacy', 'download', 'en_core_web_md'])
        install.run(self)

setup(
    cmdclass={
        'install': PostInstallCommand,
    }
)