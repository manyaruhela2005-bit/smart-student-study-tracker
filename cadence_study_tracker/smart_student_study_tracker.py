import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Set writable cache/config directories before importing pyplot.
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib.pyplot as plt  # noqa: E402


DATASET_PATH = Path("/Users/manyaruhela/Desktop/student_performance_prediction.csv")
PLOTS_FOLDER = Path.cwd() / "study_tracker_outputs"
CACHE_FOLDER = Path.cwd() / ".cache"
MPL_FOLDER = Path.cwd() / ".matplotlib"


class SmartStudentStudyTracker:
    """
    Console-based project that combines:
    1. Study session tracking
    2. Multi-day study record collection
    3. Data cleaning and preprocessing
    4. Linear Regression model training
    5. Data visualization
    6. Final score prediction for new student inputs
    """

    def __init__(self):
        self.study_records = []
        self.focus_start_time = None
        self.best_session_hours = 0.0
        self.model = None
        self.preprocessor = None
        self.feature_columns = [
            "study_hours_per_week",
            "attendance_rate",
            "previous_grades",
            "participation_in_extracurricular_activities",
            "parent_education_level",
        ]
        self.target_column = "final_performance"
        self.cleaned_df = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        PLOTS_FOLDER.mkdir(exist_ok=True)
        CACHE_FOLDER.mkdir(exist_ok=True)
        MPL_FOLDER.mkdir(exist_ok=True)

    def print_header(self):
        print("\n" + "=" * 72)
        print("SMART STUDENT STUDY TRACKER AND PERFORMANCE PREDICTION SYSTEM")
        print("=" * 72)

    def print_menu(self):
        print("\nChoose an option:")
        print("1. Start focus mode study session")
        print("2. Add a study record manually")
        print("3. View collected study records")
        print("4. Run data preprocessing, visualization, and model training")
        print("5. Predict final score for new input values")
        print("6. Exit")

    def get_float_input(self, prompt, minimum=0.0, maximum=None):
        """
        Repeatedly ask the user for a numeric value until valid input is given.
        """
        while True:
            user_input = input(prompt).strip()
            try:
                value = float(user_input)
                if value < minimum:
                    print(f"Please enter a value greater than or equal to {minimum}.")
                    continue
                if maximum is not None and value > maximum:
                    print(f"Please enter a value less than or equal to {maximum}.")
                    continue
                return value
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    def get_day_label(self):
        """
        Automatically label records as Day 1, Day 2, Day 3, and so on.
        """
        return f"Day {len(self.study_records) + 1}"

    def start_focus_mode(self):
        """
        Start a real study session using the time module.
        The user presses Enter to start and Enter again to stop.
        """
        print("\nFOCUS MODE")
        print("Press Enter to start your study session.")
        input()

        self.focus_start_time = time.time()
        print("Study session started.")
        print("Press Enter when you want to stop the session.")
        input()

        end_time = time.time()
        total_seconds = end_time - self.focus_start_time
        study_hours = total_seconds / 3600

        print(f"\nSession completed. Total study time: {study_hours:.4f} hours")

        attendance = self.get_float_input("Enter attendance percentage for this day: ", 0, 100)
        previous_score = self.get_float_input("Enter previous score: ", 0, 100)

        self.add_record(
            day=self.get_day_label(),
            study_hours=study_hours,
            attendance=attendance,
            previous_score=previous_score,
            source="Focus Mode",
        )

    def add_manual_record(self):
        """
        Add study data manually without using the timer.
        This is useful for demonstration and faster academic testing.
        """
        print("\nADD MANUAL STUDY RECORD")
        study_hours = self.get_float_input("Enter study hours for the day: ", 0)
        attendance = self.get_float_input("Enter attendance percentage: ", 0, 100)
        previous_score = self.get_float_input("Enter previous score: ", 0, 100)

        self.add_record(
            day=self.get_day_label(),
            study_hours=study_hours,
            attendance=attendance,
            previous_score=previous_score,
            source="Manual Entry",
        )

    def add_record(self, day, study_hours, attendance, previous_score, source):
        """
        Store study records in a structured list of dictionaries.
        This list can later be converted into a pandas DataFrame.
        """
        record = {
            "Day": day,
            "Study Hours": round(study_hours, 4),
            "Attendance Percentage": round(attendance, 2),
            "Previous Score": round(previous_score, 2),
            "Entry Source": source,
        }
        self.study_records.append(record)

        print("\nStudy record added successfully.")
        self.check_milestone(study_hours)

    def check_milestone(self, study_hours):
        """
        Milestone system that detects the highest study session.
        """
        if study_hours > self.best_session_hours:
            self.best_session_hours = study_hours
            print(
                "Milestone achieved! This is your highest study session so far."
            )
            print(
                f"Congratulations! New personal best: {self.best_session_hours:.4f} hours"
            )

    def view_study_records(self):
        """
        Show the study records collected so far.
        """
        if not self.study_records:
            print("\nNo study records available yet.")
            return

        records_df = pd.DataFrame(self.study_records)
        print("\nCOLLECTED STUDY RECORDS")
        print(records_df.to_string(index=False))

    def load_and_prepare_dataset(self):
        """
        Load the dataset and perform data cleaning and preprocessing.
        The original dataset does not include a direct final numeric score,
        so a realistic 'final_performance' score is created for regression.
        """
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

        df = pd.read_csv(DATASET_PATH)

        print("\nOriginal dataset shape:", df.shape)

        # Standardize column names to make the code easier to read.
        df.columns = [
            column.strip().lower().replace(" ", "_")
            for column in df.columns
        ]

        # Remove duplicate rows.
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates().copy()
        print("Duplicate rows removed:", duplicates_before)

        # Convert numeric columns to correct data types.
        numeric_columns = [
            "study_hours_per_week",
            "attendance_rate",
            "previous_grades",
        ]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Clean text columns and make them consistent.
        text_columns = [
            "participation_in_extracurricular_activities",
            "parent_education_level",
            "passed",
        ]
        for column in text_columns:
            df[column] = df[column].astype("string").str.strip()
            df[column] = df[column].replace({"nan": pd.NA, "None": pd.NA})

        # Fill missing numeric values with the median.
        for column in numeric_columns:
            df[column] = df[column].fillna(df[column].median())

        # Fill missing text values with the mode.
        for column in text_columns:
            mode_value = df[column].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)

        # Convert yes/no text values into small performance adjustments.
        participation_bonus = (
            df["participation_in_extracurricular_activities"]
            .str.lower()
            .map({"yes": 4, "no": 0})
            .fillna(0)
        )
        passed_bonus = (
            df["passed"]
            .str.lower()
            .map({"yes": 8, "no": -4})
            .fillna(0)
        )

        # Create a realistic numeric target for Linear Regression.
        # This is useful because the provided dataset has "Passed" instead of a final score.
        df[self.target_column] = (
            0.45 * df["previous_grades"]
            + 1.10 * df["study_hours_per_week"]
            + 0.25 * df["attendance_rate"]
            + participation_bonus
            + passed_bonus
        ).clip(0, 100)

        # Keep only the columns needed for the project workflow.
        prepared_df = df[self.feature_columns + [self.target_column]].copy()
        self.cleaned_df = prepared_df

        print("Cleaned dataset shape:", prepared_df.shape)
        print("\nFirst five cleaned rows:")
        print(prepared_df.head().to_string(index=False))

        return prepared_df

    def train_model(self):
        """
        Split the data into training and testing sets, preprocess features,
        and train a Linear Regression model.
        """
        df = self.load_and_prepare_dataset()

        X = df[self.feature_columns]
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        numeric_features = [
            "study_hours_per_week",
            "attendance_rate",
            "previous_grades",
        ]
        categorical_features = [
            "participation_in_extracurricular_activities",
            "parent_education_level",
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("\nMODEL TRAINING COMPLETED")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples : {len(X_test)}")
        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R^2  : {r2:.4f}")

        return y_test, y_pred

    def create_visualizations(self):
        """
        Create all required graphs:
        1. Days vs Study Hours
        2. Study Hours vs Final Performance
        3. Actual vs Predicted Values
        """
        if self.cleaned_df is None or self.model is None:
            self.train_model()

        # Graph 1: Days vs Study Hours
        if self.study_records:
            records_df = pd.DataFrame(self.study_records)
        else:
            # Add sample records automatically if the user has not entered any.
            records_df = pd.DataFrame(
                [
                    {"Day": "Day 1", "Study Hours": 2.5},
                    {"Day": "Day 2", "Study Hours": 3.0},
                    {"Day": "Day 3", "Study Hours": 1.8},
                    {"Day": "Day 4", "Study Hours": 4.2},
                ]
            )
            print(
                "\nNo manual study records were found, so sample daily records were used "
                "for the Days vs Study Hours graph."
            )

        plt.figure(figsize=(8, 5))
        plt.bar(records_df["Day"], records_df["Study Hours"], color="skyblue")
        plt.title("Days vs Study Hours")
        plt.xlabel("Days")
        plt.ylabel("Study Hours")
        plt.tight_layout()
        days_plot = PLOTS_FOLDER / "days_vs_study_hours.png"
        plt.savefig(days_plot)
        plt.show()
        plt.close()

        # Graph 2: Study Hours vs Final Performance
        plt.figure(figsize=(8, 5))
        plt.scatter(
            self.cleaned_df["study_hours_per_week"],
            self.cleaned_df[self.target_column],
            color="green",
            alpha=0.5,
        )
        plt.title("Study Hours vs Final Performance")
        plt.xlabel("Study Hours per Week")
        plt.ylabel("Final Performance")
        plt.tight_layout()
        scatter_plot = PLOTS_FOLDER / "study_hours_vs_final_performance.png"
        plt.savefig(scatter_plot)
        plt.show()
        plt.close()

        # Graph 3: Actual vs Predicted Values
        plt.figure(figsize=(8, 5))
        plt.scatter(self.y_test, self.y_pred, color="orange", alpha=0.5)
        min_value = min(self.y_test.min(), self.y_pred.min())
        max_value = max(self.y_test.max(), self.y_pred.max())
        plt.plot([min_value, max_value], [min_value, max_value], color="red")
        plt.title("Actual vs Predicted Final Performance")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.tight_layout()
        actual_predicted_plot = PLOTS_FOLDER / "actual_vs_predicted.png"
        plt.savefig(actual_predicted_plot)
        plt.show()
        plt.close()

        print("\nVISUALIZATIONS CREATED SUCCESSFULLY")
        print(f"Saved: {days_plot}")
        print(f"Saved: {scatter_plot}")
        print(f"Saved: {actual_predicted_plot}")

    def predict_new_score(self):
        """
        Accept new feature values from the user and predict the final score.
        """
        if self.model is None:
            print("\nModel not trained yet. Training the model now...")
            self.train_model()

        print("\nPREDICT FINAL PERFORMANCE")
        study_hours = self.get_float_input("Enter study hours per week: ", 0)
        attendance = self.get_float_input("Enter attendance percentage: ", 0, 100)
        previous_score = self.get_float_input("Enter previous score: ", 0, 100)

        extracurricular = input(
            "Participation in extracurricular activities (Yes/No): "
        ).strip().title()
        if extracurricular not in {"Yes", "No"}:
            extracurricular = "No"

        parent_education = input(
            "Parent education level (High School/Associate/Bachelor/Master/PhD): "
        ).strip().title()
        if not parent_education:
            parent_education = "Unknown"

        new_data = pd.DataFrame(
            [
                {
                    "study_hours_per_week": study_hours,
                    "attendance_rate": attendance,
                    "previous_grades": previous_score,
                    "participation_in_extracurricular_activities": extracurricular,
                    "parent_education_level": parent_education,
                }
            ]
        )

        prediction = float(self.model.predict(new_data)[0])
        prediction = max(0, min(100, prediction))

        print(f"\nPredicted Final Score: {prediction:.2f}")

    def run_full_workflow(self):
        """
        Run the full data science workflow in one option.
        """
        self.train_model()
        self.create_visualizations()

    def run(self):
        """
        Main console loop for the project.
        """
        self.print_header()
        print(f"Dataset path: {DATASET_PATH}")

        while True:
            self.print_menu()
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                self.start_focus_mode()
            elif choice == "2":
                self.add_manual_record()
            elif choice == "3":
                self.view_study_records()
            elif choice == "4":
                self.run_full_workflow()
            elif choice == "5":
                self.predict_new_score()
            elif choice == "6":
                print("\nThank you for using the system. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 6.")


if __name__ == "__main__":
    tracker = SmartStudentStudyTracker()
    tracker.run()
