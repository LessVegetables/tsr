from pathlib import Path

import pandas as pd

from upsetkit import ConditionStyle, SubsetRule, UpsetStyle, plot_upset

HERE = Path(__file__).parent
df = pd.read_csv(HERE.parent / "example-test-Titanic-Dataset.csv")  # it is actually a CSV

# 1. Which columns tend to be missing together?
missing = plot_upset(
    df,
    {col: f"{col}.isna()" for col in ["Age", "Cabin", "Embarked"]},
    UpsetStyle(title="Missing values", intersection_bar_color="#6c5ce7", totals_bar_color="#a29bfe"),
)
missing.figure.savefig(HERE / "titanic_missing.png", bbox_inches="tight")

# 2. How do survival, sex, age and class overlap? Bars at the bottom, conditions on the right.
survival = plot_upset(
    df,
    {
        "Survived": "Survived",
        "Male": "Sex == 'male'",
        "Child": "Age < 18",
        "First class": "Pclass == 1",
        "Travelling alone": lambda d: (d["SibSp"] + d["Parch"]) == 0,
    },
    UpsetStyle(
        intersections_side="top",
        conditions_side="left",
        title="Titanic passengers",
        min_subset_size=5,
        sort_by="cardinality",
        background_color="#fafafa",
        inactive_dot_color="#e0e0e0",
        condition_styles={"Survived": ConditionStyle(totals_bar_color="#2e7d32", label_color="#2e7d32")},
        subset_rules=[
            SubsetRule(present=["Survived"], bar_color="#2e7d32", active_dot_color="#2e7d32",
                       line_color="#2e7d32", label="Survived"),
            SubsetRule(absent=["Survived"], bar_color="#c62828", label="Died"),
        ],
    ),
)
survival.figure.savefig(HERE / "titanic_survival.png", bbox_inches="tight")
print(survival.intersections)
