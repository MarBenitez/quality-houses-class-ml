# House Quality Prediction in Switzerland

This project focuses on predicting the quality of houses in Switzerland using machine learning and data analysis techniques. The goal is to provide insights into house quality and enable users to predict the quality of their own homes based on specific input values.

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data Analysis](#data-analysis)
- [Machine Learning Workflow](#machine-learning-workflow)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Data Cleaning](#data-cleaning)
  - [Feature Engineering](#feature-engineering)
  - [Clustering](#clustering)
  - [Classification](#classification)
- [Streamlit Application](#streamlit-application)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

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

## Data Analysis
A thorough Exploratory Data Analysis (EDA) was conducted to understand the dataset better, identify patterns, and detect anomalies. Data cleaning was performed to handle missing values, outliers, and other inconsistencies.

## Machine Learning Workflow

### Exploratory Data Analysis (EDA)
EDA involved visualizing the data distribution, understanding relationships between variables, and identifying key features impacting house quality.

### Data Cleaning
Data cleaning processes included handling missing values, removing duplicates, and correcting data types to ensure the dataset was ready for analysis.

### Feature Engineering
Feature engineering was performed to create new features from existing data, enhancing the model's predictive power. This included creating a new column representing the sum of all columns related to street views.

### Clustering
Clustering techniques were used to group similar houses together, providing insights into different categories of house quality.

### Classification
Classification models were built to predict house quality. Various algorithms were tested, and the best-performing model was selected.

## Streamlit Application
A Streamlit application was developed to visualize the results of the analysis and predictions. The application allows users to input their house details and get a predicted quality score.

## Usage
To use the Streamlit application:
1. Clone this repository.
2. Install the necessary dependencies.
3. Run the Streamlit application and input your house details to get a quality prediction.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-quality-prediction.git
