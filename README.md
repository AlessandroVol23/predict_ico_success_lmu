# data_science_for_business_lmu

## Next Deadline 

__13.12.2019__

## Paper

### Outline
1. Introduction
    1. ICO
    2. Machine Learning - Motivation
    3. ICO - ML - Motvation
2. Approach
3. Data Analysis
    1. Description of data
    2. EDA
    3. "Statistical Analysis" -> Correlations / Min / Max / Mean
4. Prediction Model
    1. Pipeline
    2. Data Preparation
    3. Prediction Model
        -> Random Forrest / Light Gbm
5. Results
6. Conclusion and Outlook


## Currently implemented 

### Data
- Data pre-generation through make command 
### Features
- Automatically generation of Features
- Automatically exporting of features into CSV files 
- Custom Feature set description through a meta file saved in JSON data structure 
- Random feature meta file generation 
- Generation of random feature sets through random feature meta file 
- NA-value Strategies [mean, median, delete, set]
- Encoder strategies [one_hote, label]

### Models
- saving model results into result file 
- automatically generating and incrementing of submission number, tied to result and feature set 
- cross-fold validation
- up-sampling
- pipeline to fit a model to random generated features 
- implemented models [LightGBM, Catboost]