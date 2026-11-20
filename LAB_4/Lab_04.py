import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

import bnlearn as bn
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# ---------------------------------------------------
# 1. READ DATA
# ---------------------------------------------------

columns = [
    "EC100", "EC160", "IT101", "IT161",
    "MA101", "PH100", "PH160", "HS101", "QP"
]

data = pd.read_csv(
    "2020_bn_nb_data.txt",
    sep=r"\s+",
    names=columns,
    skiprows=1
)

print("Dataset shape:", data.shape)
print(data.head())

grade_categories = ["F", "DD", "CD", "CC", "BC", "BB", "AB", "AA"]
course_columns = columns[:-1]
qp_categories = sorted(data["QP"].unique())

full_state_names = {
    col: grade_categories
    for col in course_columns
}
full_state_names["QP"] = qp_categories


# ---------------------------------------------------
# 2. LEARN COURSE DEPENDENCIES USING CHOW-LIU
# ---------------------------------------------------

course_data = data.drop(columns=["QP"])

model_struct = bn.structure_learning.fit(
    course_data,
    methodtype="chow-liu",
    verbose=0
)

print("\nLearned Dependency Tree Edges:")
print(model_struct["model_edges"])


# ---------------------------------------------------
# 3. LEARN BAYESIAN NETWORK PARAMETERS
# ---------------------------------------------------

model = bn.parameter_learning.fit(
    model_struct,
    course_data,
    methodtype="bayes",
    verbose=0
)


# ---------------------------------------------------
# 4. DISPLAY CONDITIONAL PROBABILITY TABLES
# ---------------------------------------------------

print("\nCPTs:")
print("--------------------------------")
bn.print_CPD(model)


# ---------------------------------------------------
# 5. INFER PH100 GRADE
# ---------------------------------------------------

query_res = bn.inference.fit(
    model,
    variables=["PH100"],
    evidence={
        "EC100": "DD",
        "IT101": "CC",
        "MA101": "CD"
    },
    verbose=0
)

print("\nPH100 Probability Table:")
print(query_res.df)

most_probable_ph100 = query_res.df.iloc[
    query_res.df["p"].idxmax()
]["PH100"]

print("\nMost probable PH100 grade:", most_probable_ph100)


# ---------------------------------------------------
# 6. PREPARE DATA FOR NAIVE BAYES
# ---------------------------------------------------

X = data.drop(columns=["QP"])
y = data["QP"]

encoder = OrdinalEncoder(
    categories=[grade_categories] * X.shape[1],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_encoded = encoder.fit_transform(X)


# ---------------------------------------------------
# 7. RUN 20 NAIVE BAYES EXPERIMENTS
# ---------------------------------------------------

nb_results = []

print("\nNAIVE BAYES RESULTS")
print("--------------------------------")

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.30,
        random_state=i,
        stratify=y
    )

    clf = CategoricalNB(min_categories=len(grade_categories))
    clf.fit(X_train.astype(int), y_train)

    accuracy = clf.score(X_test.astype(int), y_test)
    nb_results.append(accuracy * 100)

    print(f"Experiment {i + 1} Accuracy = {accuracy * 100:.2f}%")

print(f"\nAverage Naive Bayes Accuracy = {np.mean(nb_results):.2f}%")
print(f"Standard Deviation = {np.std(nb_results):.2f}%")


# ---------------------------------------------------
# 8. RUN 20 DEPENDENCY-AWARE TAN EXPERIMENTS
# ---------------------------------------------------

tan_results = []

print("\nDEPENDENCY-AWARE CLASSIFIER")
print("--------------------------------")

for i in range(20):
    train, test = train_test_split(
        data,
        test_size=0.30,
        random_state=i,
        stratify=data["QP"]
    )

    train_courses = train.drop(columns=["QP"])

    # Learn the course dependency tree from this training split.
    struct = bn.structure_learning.fit(
        train_courses,
        methodtype="chow-liu",
        verbose=0
    )

    # Attach QP as a parent of every course variable.
    edges = list(struct["model_edges"])
    for course in course_columns:
        edges.append(("QP", course))

    tan_bn = DiscreteBayesianNetwork(edges)

    estimator = BayesianEstimator(
        tan_bn,
        train,
        state_names=full_state_names
    )

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=10
    )

    tan_bn.add_cpds(*cpds)
    infer = VariableElimination(tan_bn)

    correct = 0

    for _, student in test.iterrows():
        evidence = {
            course: student[course]
            for course in course_columns
        }

        result = infer.query(
            variables=["QP"],
            evidence=evidence,
            show_progress=False
        )

        predicted = result.state_names["QP"][
            np.argmax(result.values)
        ]

        if predicted == student["QP"]:
            correct += 1

    accuracy = (correct / len(test)) * 100
    tan_results.append(accuracy)

    print(f"Experiment {i + 1} Accuracy = {accuracy:.2f}%")


# ---------------------------------------------------
# 9. FINAL RESULTS
# ---------------------------------------------------

print("\n================================")
print("FINAL RESULTS")
print("================================")
print(f"Naive Bayes Average Accuracy : {np.mean(nb_results):.2f}%")
print(f"Naive Bayes Std Deviation    : {np.std(nb_results):.2f}%")
print(f"Dependency-Aware Average     : {np.mean(tan_results):.2f}%")
print(f"Dependency-Aware Std Dev     : {np.std(tan_results):.2f}%")
