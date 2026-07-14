# Cadence Student Study Tracker

Cadence is a student study tracker and performance prediction project. It combines a clean browser-based frontend with a Python console workflow for study session tracking, data preprocessing, machine learning, and visualization.

The project is designed for students who want to record study habits, review progress, and estimate final academic performance from study hours, attendance, previous grades, extracurricular activity, and parent education level.

## Features

- Browser-based study tracker UI
- Focus mode timer for study sessions
- Manual study record entry
- Study records dashboard and ledger
- Analytics charts for study activity and model results
- Final performance prediction screen
- Python console version with data cleaning, Linear Regression, and graph generation
- Double-clickable macOS app launcher

## Project Preview

Cadence includes three generated visualizations:

- Days vs Study Hours
- Study Hours vs Final Performance
- Actual vs Predicted Final Performance

The saved graph images are available in:

```text
study_tracker_outputs/
```

## Project Structure

```text
cadence_study_tracker/
├── study_tracker_outputs/           # Generated graph images
├── cadence.html                     # Main browser study tracker app
├── index.html                       # Landing page
├── smart_student_study_tracker.py   # Python console tracker and ML workflow
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

## How to Run the Frontend

### Option 1: Open Like a Mac App

Double-click:

```text
Cadence Study Tracker.app
```

If macOS blocks it because it is unsigned, right-click the app, choose **Open**, then choose **Open** again.

### Option 2: Open in a Browser

Open `index.html` in your browser, then click **Open Cadence**.

You can also open `cadence.html` directly.

### Option 3: Use VS Code Live Server

1. Open the project folder in VS Code.
2. Install the **Live Server** extension if you do not already have it.
3. Right-click `index.html`.
4. Select **Open with Live Server**.

## How to Run the Python Console App

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the console program:

```bash
python smart_student_study_tracker.py
```

The Python app provides this menu:

```text
1. Start focus mode study session
2. Add a study record manually
3. View collected study records
4. Run data preprocessing, visualization, and model training
5. Predict final score for new input values
6. Exit
```

## Dataset Requirement

The Python workflow expects the dataset CSV file at:

```text
/Users/manyaruhela/Desktop/student_performance_prediction.csv
```

The dataset should include columns similar to:

```text
study_hours_per_week
attendance_rate
previous_grades
participation_in_extracurricular_activities
parent_education_level
passed
```

The script standardizes column names, removes duplicate rows, fills missing values, creates a numeric `final_performance` target, trains a Linear Regression model, and saves graphs.

## Machine Learning Workflow

The Python app performs the following steps:

1. Loads the student performance dataset.
2. Cleans column names and removes duplicates.
3. Converts numeric columns safely.
4. Fills missing numeric values with medians.
5. Fills missing text values with modes.
6. Creates a numeric final performance score.
7. Splits the data into training and testing sets.
8. Trains a Linear Regression model.
9. Evaluates the model using MAE, MSE, RMSE, and R².
10. Generates and saves visualizations.

## Technologies Used

- HTML
- CSS
- JavaScript
- Chart.js
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Notes

- The browser UI runs locally and stores study records only during the current browser session.
- The analytics charts in the frontend use Chart.js from a CDN, so internet access may be needed when loading charts for the first time.
- The macOS app launcher simply opens the local `cadence.html` file.
- The Python console app and browser UI are separate implementations of the same study tracker idea.

## Author

Created by Manya Ruhela.
