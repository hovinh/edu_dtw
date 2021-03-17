# edu_dtw
An education toolkit to illustrate Dynamic Time Warping algorithm. Thie materials originally prepared for this <a href="https://hovinh.github.io/blog/">blog post</a>. You can create virtual environment as below and jump straight to the `tutorial.ipynb`. 

## Environment Setup

You create a virtual environment for this project by entering Anaconda Prompt, navigating to this directory:
```bash
conda env create -f environment.yml
```
Once succeed, you have a new conda virtual env named "edu_dtw". Alternatively, one can install them manually:

```bash
conda create -n edu_dtw python=3.6
conda activate edu_dtw
conda install numpy
conda install -c conda-forge matplotlib
conda install -c anaconda seaborn
conda install -c anaconda scipy
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=edu_dtw
```

## Getting-started
You are ready to run the code in `tutorial.ipynb` with a kernel named `edu_dtw`.


## Contact
Feel free to contact me via email: hxvinh.hcmus@gmail.com
