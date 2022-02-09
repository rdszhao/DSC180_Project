# DSC180 Capstone Project

Most of the project code was actually built using Jupyter Notebooks, so the latest working code would be found in the `Notebooks` folder. Here the latest figures and hyperparameter tuning can be found. Demonstrations of the Neural Net, Support Vector Machine, and Random Forest are here.

The code is run via the command `python run.py test`. This runs the baseline model on the test data, which is simply the normal data but randomized.

Project code with working models can be found in `src` folder. Here the code for data generation and the test data can be found. This is also where the model code can be found which is in the `prediction` folder. Here, the code for the feed-forward Neural Net and the SVM can be found.

To build the `docker build -t <tag_name> .` which gives a local docker container with the libraries scipy, numpy, pandas, pytorch, and sklearn.
