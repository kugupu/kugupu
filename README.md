<img src="docs/kugupu_logo.png" width="400">

## **kgp - a molecular network generator to study charge transport pathways in amorphous materials** 


kgp is a package for the postprocessing of molecular dynamics trajectories of organic semiconductors. It is built on MDAnalysis, NetworkX and YAeHMOP.


### Installation instructions

Installation is best done using the conda env file

```bash
git clone https://github.com/richardjgowers/kugupu.git
cd kugupu
# install requirements
conda env create -f kgp_env.yml
# install the kugupu package
pip install .
cd ../

# Clone and install the Python bindings to yaehmop
# This will get streamlined later...
git clone https://github.com/richardjgowers/pyyaehmop.git
cd yaehmop
pip install .
cd ../
```

### Example notebooks

```bash
cd kugupu/notebooks

jupyter notebook

```


### Copyright

Copyright (c) 2018, Micaela Matta and Richard J Gowers


#### Acknowledgements
 
Project based on the 
[Computational Chemistry Python Cookiecutter](https://github.com/choderalab/cookiecutter-python-comp-chem)
