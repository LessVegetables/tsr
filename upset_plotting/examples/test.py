from upsetkit import plot_upset, UpsetStyle, SubsetRule, ConditionStyle

result = plot_upset(
    df,
    {"Survived": "Survived",                 # an existing 0/1 column
     "Male": "Sex == 'male'",                # an expression
     "Child": "Age < 18",
     "Alone": lambda d: d.SibSp + d.Parch == 0},  # a function
    UpsetStyle(
        intersections_side="bottom", conditions_side="right",
        intersection_bar_color="gray",
        condition_styles={"Survived": ConditionStyle(totals_bar_color="green")},
        subset_rules=[SubsetRule(present=["Survived"], bar_color="green", label="Survived")],
    ),
)
result.figure.savefig("plot.png")