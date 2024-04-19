# Probabilistic Residential Energy Demand Forecasting Pipeline (New York)

### Author: Tobias Butler

This Repository contains Python modules and Jupyter notebooks that can be used to create a probabilistic energy demand forecasting pipeline. More specifically, one that predicts the hourly residential energy demand in New York City, NY. This pipeline was created as the capstone project for my Masters program through Grand Canyon University.

# User Guide

There are two predominant ways to use this project. First, one can see all exploratory and analytical results presented through a series of visuals and results tables in the [project's web application](https://energydemandforecasting-2.onrender.com/). The link will take users to the app's homepage where they are presented with a user guide for the application.

The second way users are intended to use this project repository is to reproduce an instance of the forecasting pipeline, adapt it, or integrate a portion of it into one's own project. The pipeline itself is split into three components: data collection, data processing, and the forecasting model. While originally designed to work sequentially, each of these components could function on its own and therefore could be extracted from this pipeline and inserted into another (with minimal tweaks required). 

In order for users to understand how each of these components works, a jupyter notebook has been created to demonstrate the creation and usage of these three components. This notebook is called "data_modeling_walkthrough.ipynb". In it, users can see an example call to the data collection component