# SDG 4 : Education
# Student Performance Prediction Using Machine Learning

## 📑 Project Description

This project uses spervised machine learning to predict student academic performance based on a variety of factors such as previous grades, study habits, absences, family background, and personal characteristics.

Leverages the **UCI Student Performance Dataset**, to show how predictive models can help educators identify students who may need additional support early on.

---
# Problem Statement

Educational institutions constantly seek ways to improve student outcomes. One major challenge is identifying students at risk of poor performance before final grades are issued.

**Goal:**  
- Build a supervised machine learning model that predicts students' final grades (`G3`) based on factors such academic, social, and behavioral attributes.

## 🚀 Tools & Technologies
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Model:** Random Forest Regressor  
- **Dataset:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)  
---

## 📈 Steps Followed

1. **Data Loading:**  
   Load the dataset (`student-mat.csv`).

2. **Data Exploration:**  
   - Preview the dataset.  
   - Identify categorical and numerical variables.  
   - Visualize data distribution ( a histogram for grades distribution, boxplots for outliers demonstrated on the screenshots)

3. **Data Preprocessing:**  
   - Encode categorical variables using one-hot encoding.  
   - Check and handle missing values (none found).  
   - Check for and handle duplicate rows.

4. **Feature and Target Definition:**  
   - `X` → Features (all other variable columns).  
   - `y` → Target variable(`G3` - final grade).

5. **Train-Test Split:**  
   Split the data into training (80%) and testing (20%).

6. **Model Training:**  
- Uses **Random Forest Regressor**, an ensemble learning method that handles nonlinear relationships.

7. **Model Prediction:**  
   Make predictions on the test set.

8. **Model Evaluation:**  
   Evaluate using:  
   - **MAE (Mean Absolute Error)**  
   - **RMSE (Root Mean Squared Error)**  
   - **R² Score**

9. **Visualization:**  
   - Plot Actual vs Predicted Grades scatter plot with a diagonal reference line.

 ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
 # Results and Interpretation
| **Mean Absolute Error** | **1.16**  | 
* On average, the predictions are off by **\~1.16 points** from the actual grades which indicates a good grading system out of 20.    
                |
| **Root Mean Squared Error**     | **1.95**  | 
* Slightly penalizes big errors more, but still below 2 — strong performance.  
                                                                        
| **R-square** | **0.81**  | ⭐️⭐️⭐️ 
* The model explains **81%** of the variance in student grades. 

# From the Scatter Plot (Actual vs Predicted Grades) comparing actual grades (G3) to predicted grades, several conclusions can be drawn:

        🔵 The data points are closely clustered along the red diagonal line, which represents perfect predictions (where actual = predicted).

        ✅ This indicates that the model performed well — for most students, the predicted final grades are close to their actual grades.

        🔴 A few points deviate from the diagonal line, indicating some underpredictions or overpredictions - which is normal.

* ✔️ Generally, students with higher actual grades were predicted accurately, and those with lower actual grades were also reasonably well captured.

🔥 Variable Influence
The Random Forest model learns patterns and relationships from the data. Based on the analysis and common patterns in this dataset, the following variables stood out:

# 🚀 Strongest Influencers:
These variables had the most direct impact on predicting G3:

    ✔️ G2 (2nd period grade) ... Students who performed well in G2 almost always had a higher G3.
        ✔️ The strongest predictor of G3. Students who performed well in G2 almost always had a higher G3.
        ✔️ This makes sense — the final exam closely follows the second period performance.

    ✔️ G1 (1st period grade) ... Also highly correlated with G3. 

# 🟨 Moderate Influencers:
✔️ Absences
🔻 Higher absences slightly decrease performance. Students who skipped more classes tended to have slightly lower final grades.

✔️ Studytime
🔻 More study time correlates with higher grades.However less influential than G1 and G2

✔️ Failures (past class failures)
🔻 Students with more past failures tend to perform worse, but this can be offset by improved G1 and G2.

# 🏡 Sociodemographic Factors:
These have some but lesser impact:

→ Medu and Fedu (Mother's and Father's education).

→ famsup, schoolsup (family or school support) — positive but minor impact.

→ romantic (whether the student has a relationship) — very minor negative impact, likely reflecting distraction.

→ famrel, goout, Dalc, Walc (family relationship, going out, weekday/weekend alcohol consumption) — have weak correlations 

🔍 Summary of How the Model Makes Decisions:

* Main formula the model implicitly learns:
→ Students with high G1 and G2, low absences, no past failures, and moderate to high study time are very likely to achieve high G3.

* Conversely:
→ Students with poor G1 and G2, high absences, and previous failures are predicted to have lower G3.

→ Even if a student had a bad G1 but improved dramatically in G2, the model adjusts accordingly — G2 has a stronger influence on G3.

