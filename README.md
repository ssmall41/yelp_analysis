# yelp_analysis
An analysis of Yelp data from Kaggle

This is an analysis of the data provided on Kaggle at:
https://www.kaggle.com/yelp-dataset/yelp-dataset

Included in this repo is:

1. A docker file for running the code.
2. A presentation summarizing the findings.
3. Data files. This repo does not contain ALL of the provided data, just that which is used.
4. A python module models.py which builds and trains the models.


To run the code that builds the models:
1. Run 'docker build -t yelp_data .'
2. Run 'docker run -it yelp_data /bin/bash' to start and enter the container.
3. The model can be run using 'python models.py'


The file presentation.pdf includes a summary of the findings from this analysis.
