import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
from IPython.display import display

plt.style.use("seaborn-v0_8")

random_state = 42


def print_all_results(num_task, name_estimator):
    df = pd.read_csv("outputs/" + str(num_task) + "_" + name_estimator + "_output.csv")
    display(
        df[[col for col in df.columns if col not in ["MSE", "R^2"]] + ["MSE", "R^2"]]
        .sort_values(by="R^2", ascending=False)
        .style.format(precision=7)
    )


def save_pickle(task, name_estimator, best_preprocessor, best_estimator):
    file = open("preprocess_saves/" + str(task) + "_" + name_estimator + "_preproc_" + ".save", "wb")
    pickle.dump(best_preprocessor, file)
    file = open("model_saves/" + str(task) + "_" + name_estimator + "_model_" + ".save", "wb")
    pickle.dump(best_estimator, file)
    file.close()


def show_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(None)
    fig.tight_layout()
    cmp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        np.round(y_pred).astype(int),
        ax=ax,
        # xticks_rotation=80,
        include_values=False,
        cmap="gist_stern",  # "magma",
        colorbar=False,
    )
    ax.set_xticks(range(0, len(cmp.confusion_matrix), 5))
    ax.set_yticks(range(0, len(cmp.confusion_matrix), 5))
    ax.plot(range(len(cmp.confusion_matrix)), range(len(cmp.confusion_matrix)), color="white", linestyle="dotted")
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    plt.colorbar(cmp.im_, cax=cax)
    plt.show()


def show_parallel_plot(num_task, name_estimator):
    # Show Parallel graph
    df = pd.read_csv("outputs/" + str(num_task) + "_" + name_estimator + "_output.csv")
    df["R^2"] = df["R^2"].round(2)
    df["MSE"] = df["MSE"].round(2)

    for col in df.columns:
        if col not in ["R^2", "MSE"]:
            df[col] = df[col].fillna("None")
            df[col] = df[col].astype("object")

    dimensions = []

    for col in df.select_dtypes(include="object"):
        if col not in ["R^2", "MSE"]:
            unique_dict = {i: num for num, i in enumerate(df[col].unique())}
            df[col] = df[col].map(unique_dict)
            dimensions.append(
                dict(
                    label=str(col),
                    values=df[col],
                    tickvals=list(unique_dict.values()),
                    ticktext=list(unique_dict.keys()),
                )
            )

    for col in ["R^2", "MSE"]:
        dimensions.append(dict(label=str(col), values=df[col]))

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=df["MSE"], colorscale="viridis", showscale=True), dimensions=dimensions, labelangle=-15
        )
    )

    fig.update_traces(
        dimensions=[
            {
                **d,
                **{
                    "tickvals": np.round(
                        np.linspace(min(d["values"]), max(d["values"]), len(np.unique(d["values"]))), 2
                    )
                },
            }
            for d in fig.to_dict()["data"][0]["dimensions"]
        ]
    )

    # fig.update_layout(
    #     paper_bgcolor="floralwhite",
    # )

    fig.show()
