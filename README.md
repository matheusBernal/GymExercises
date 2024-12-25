# Gym Member Exercise Analysis and Prediction

## Description

This project analyzes gym members' exercise data to predict the number of calories burned and classify their experience levels. It uses machine learning techniques to gain insights from the data and build predictive models.

## Installation

To run this project in Google Colab, you'll need to install the following Python libraries:

bash pip install pandas==2.1.4 pip install numpy==1.26.3 pip install matplotlib==3.8.2 pip install seaborn==0.13.1 pip install sklearn==1.3.2

You must ensure that you are using these versions.

## Usage

1.  **Data Loading:** The project begins by loading the `gym_members_exercise_tracking.csv` dataset into a pandas DataFrame. Ensure that this file is uploaded to your Colab environment.
2.  **Data Preprocessing:**
    *   The `Gender` and `Workout_Type` columns are encoded using `LabelEncoder`.
    *   `OneHotEncoder` and `ColumnTransformer` are used to handle categorical variables for experience level classification.
    *   `StandardScaler` is applied to scale the features.
3.  **Regression Model (Calories Burned):**
    *   A `LinearRegression` model and `Ridge` model are trained to predict calories burned based on 'Session_Duration (hours)', 'Fat_Percentage', and 'Workout_Frequency (days/week)'.
    *   The code calculates and displays the coefficients, intercept, and training/testing scores for the models.
4.  **Classification Models (Experience Level):**
    *   The project trains and evaluates multiple classification models:
        *   `GaussianNB`
        *   `DecisionTreeClassifier`
        *   `RandomForestClassifier`
        *   `SVC`
        *   `KNeighborsClassifier`
        *   `MLPClassifier` (Neural Network)
    *   `GridSearchCV` is used to find the best hyperparameters for each model.
    *   The code includes functions to calculate accuracy, precision, recall, and F1-score.
    *   `classification_report` is used to get a more detailed evaluation.
5. **Final Dataframe**
    * A dataframe will be created and saved with all the predictions of each model. The user can find the file named `gym_members_exercise_tracking_final.xlsx` in the files section of the environment.

## Data

The project uses the `gym_members_exercise_tracking.csv` dataset, which includes the following features:

*   Age
*   Gender
*   Weight (kg)
*   Height (m)
*   Max_BPM
*   Avg_BPM
*   Resting_BPM
*   Session_Duration (hours)
*   Fat_Percentage
*   Water_Intake (liters)
*   Workout_Frequency (days/week)
*   BMI
*   Workout_Type
*   Calories_Burned
*   Experience_Level

## Models

### Regression

*   **Linear Regression:** Predicts calories burned based on session duration, fat percentage, and workout frequency.
*   **Ridge:** Similar to linear regression but with L2 regularization.

### Classification

*   **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes' theorem.
*   **Decision Tree:** A tree-based model that splits the data based on feature values.
*   **Random Forest:** An ensemble of decision trees.
*   **Support Vector Machine (SVM):** A classifier that finds the optimal hyperplane to separate classes.
*   **K-Nearest Neighbors (KNN):** A non-parametric method that classifies based on proximity to neighboring points.
*   **Multi-layer Perceptron (MLP):** A neural network model.

## Results

### Regression

*   The Linear Regression model had a train score of 82.2% and a test score of 85.6%.
*   The Ridge model had a train score of 82.2% and a test score of 85.5%.

### Classification

The following table summarizes the accuracy of each classification model:

| Model               | Train Accuracy | Test Accuracy |
| ------------------- | -------------- | ------------- |
| GaussianNB          | 89.4%          | 86.6%         |
| Decision Tree       | 93.8%          | 87.7%         |
| Random Forest       | 99.1%          | 87.3%         |
| SVM                 | 89.1%          | 84.9%         |
| K-Nearest Neighbors | 86.6%          | 79.8%         |
| Neural Network      | 91.2%          | 83.6%         |

## Author

\Matheus Bernal
