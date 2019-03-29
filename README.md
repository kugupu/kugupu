kugupu
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/kugupu.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/kugupu)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/kugupu/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/kugupu/branch/master)

KUGUPUUU!

### Installation instructions

Installation is best done using the conda env file

```bash
git clone https://github.com/chapmanlab/kugupu.git
cd kugupu
# install requirements
conda env create -f kgp_env.yml
# install the kugupu package
pip install .
cd ../

# Clone and install the Python bindings to yaehmop
# This will get streamlined later...
git clone https://github.com/chapmanlab/pyyaehmop.git
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
