# to install in develop mode:> python setup.py develop
# to uninstall:> python setup.py --uninstall
from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='PySEC',
   version='0.0',
   description='python repo for spectral exterior calculus',
   long_description=long_description,
   license='Undecided',
   author='Aaron Mahler',
   author_email='aaron.mahler@teledyne.com',
   url='https://github.com/Center-for-Math-AI/PySEC',
   packages=['PySEC'],  #same as name
   install_requires=['wheel', 
                     'torch', 
                     'scipy', 
                     'numpy'], #external packages as dependencies
)
