# House Quality Prediction in Switzerland

![banner](https://keystoneacademic-res.cloudinary.com/image/upload/v1696411312/articles/educationscom/article-hero-image-4239.jpg)

This project focuses on predicting the quality of houses in Switzerland using machine learning and data analysis techniques. The goal is to provide insights into house quality and enable users to predict the quality of their own homes based on specific input values.

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Data Cleaning](#data-cleaning)
  - [Feature Engineering](#feature-engineering)
  - [Clustering](#clustering)
  - [Classification](#classification)
  - [Streamlit Application](#streamlit-application)
    - [Usage](#usage)

### _Note_
_The original dataset used in this project is very large, which prevents us from uploading it directly to this GitHub repository. You can access the dataset through the following link: [Swiss Houses Dataset on Zenodo](https://zenodo.org/records/7070952#.Y0mACy0RqO0)._

## Introduction
This project utilizes various machine learning and data analysis techniques to predict the quality of houses in Switzerland. The dataset contains various attributes describing different aspects of houses, and the objective is to build a model that can accurately predict house quality.

## Project Overview
The project includes the following key steps:
1. Exploratory Data Analysis (EDA)
2. Data Cleaning
3. Feature Engineering
4. Clustering
5. Classification
6. Streamlit Application for visualization and prediction

### 1. Exploratory Data Analysis (EDA)
EDA involved visualizing the data distribution, understanding relationships between variables, and identifying key features impacting house quality.

### 2. Data Cleaning
Data cleaning processes included handling missing values, removing duplicates, and correcting data types to ensure the dataset was ready for analysis.

### 3. Feature Engineering
Feature engineering was performed to create new features from existing data, enhancing the model's predictive power. This included creating a new column representing the sum of all columns related to street views.

### 4. Clustering
Clustering techniques were used to group similar houses together, providing insights into different categories of house quality.

### 5. Classification
Classification models were built to predict house quality. Various algorithms were tested, and the best-performing model was selected.

### 6. Streamlit Application
A Streamlit application was developed to visualize the results of the analysis and predictions. The application allows users to input their house details and get a predicted quality score.

#### Usage
To use the Streamlit:

1. Your local:
   
- Clone this repository.
- Install the necessary dependencies.
- Run the Streamlit application and input your house details to get a quality prediction.

2. Link to the app:

  - [App](https://quality-houses-class-ml.streamlit.app/)
