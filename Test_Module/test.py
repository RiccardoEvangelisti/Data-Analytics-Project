import pickle
import warnings
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import torch
from torch.utils.data import DataLoader
from load_helper import MyDataset, NeuralNetworkDropoutBatchNorm

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

MY_UNIQUE_ID = "citeristi"

MODELS_DIR = "model_saves/"
PREPROCESS_DIR = "preprocess_saves/"

LR_NAME = "LR"
RF_NAME = "RF"
KNR_NAME = "KNR"
SVR_NAME = "SVR"
FF_NAME = "FF"
TB_NAME = "TB"
TF_NAME = "TF"


def get_num_task(clfName):
    if clfName in [LR_NAME, RF_NAME, KNR_NAME, SVR_NAME]:
        return "1_"
    elif clfName in [FF_NAME]:
        return "2_"
    elif clfName in [TB_NAME, TF_NAME]:
        return "3_"
    else:
        raise ValueError(
            f"Invalid classifier name. Possible values are {LR_NAME}, {RF_NAME}, {KNR_NAME}, {SVR_NAME}, {FF_NAME}, {TB_NAME}, {TF_NAME}"
        )


# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


# Input: Test dataframe
# Output: PreProcessed test dataframe
def preprocess(df, clfName):
    X = df.drop(columns=["Year"]).reset_index(drop=True)
    y = df["Year"].reset_index(drop=True)
    pipe_transformers = pickle.load(
        open(PREPROCESS_DIR + get_num_task(clfName) + clfName + "_preproc_" + ".save", "rb")
    )
    if clfName in [TB_NAME, TF_NAME]:
        X = pd.DataFrame(pipe_transformers.transform(X))
    else:
        X = pd.DataFrame(pipe_transformers.transform(X.values))
    dfNew = pd.concat([y,X], axis=1)
    return dfNew


# Input: Regressor name
# Output: Regressor object
def load(clfName):
    return pickle.load(open(MODELS_DIR + get_num_task(clfName) + clfName + "_model_" + ".save", "rb"))


# Input: PreProcessed dataset, Regressor Name, Regressor Object
# Output: Performance dictionary
def predict(df, clfName, clf):
    X = df.drop(columns=["Year"])
    y = df["Year"]

    if clfName in [LR_NAME, RF_NAME, KNR_NAME, SVR_NAME]:
        y_pred = clf.predict(X.values)
    # ----------------------------------------------------------------
    elif clfName in [FF_NAME]:

        test_dataset = MyDataset(X.values, y)
        test_loader = DataLoader(test_dataset, batch_size=1)
        with torch.no_grad():
            clf.eval()
            y_pred = []
            for data, _ in test_loader:
                y_pred.append(clf(data))
        y_pred = torch.stack(y_pred).squeeze().detach().numpy()
    # ----------------------------------------------------------------
    elif clfName in [TB_NAME, TF_NAME]:
        y_pred = clf.predict(X)
    # ----------------------------------------------------------------
    else:
        raise ValueError(
            f"Invalid classifier name. Possible values are {LR_NAME}, {RF_NAME}, {KNR_NAME}, {SVR_NAME}, {FF_NAME}, {TB_NAME}, {TF_NAME}"
        )

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    perf = {"mse": mse, "mae": mae, "mape": mape, "r2score": r2}

    return perf



# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# DA RIMUOVERE
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
from sklearn.model_selection import train_test_split

random_state = 42

df = pd.read_csv("../Train_Module/train.csv").drop_duplicates()

_, test = train_test_split(df, stratify=df["Year"], test_size=0.3, random_state=random_state)
_, test = train_test_split(test, stratify=test["Year"], test_size=1 / 3, random_state=random_state)

print(test.shape)

CLF_NAME_LIST = [LR_NAME, RF_NAME, KNR_NAME, SVR_NAME, FF_NAME, TB_NAME, TF_NAME]

for modelName in CLF_NAME_LIST:
    dfProcessed = preprocess(test, modelName)
    print(dfProcessed.shape)
    clf = load(modelName)
    perf = predict(dfProcessed, modelName, clf)
    print("RESULT team: " + str(getName()) + " algoName: " + modelName + " perf: " + str(perf))