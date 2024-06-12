import scanpy as sc
import pandas as pd
import scmags as sm
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, roc_curve, auc
import squidpy as sq
from sklearn.model_selection import cross_val_score
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import squidpy as sq
from mapie.metrics import (
    classification_coverage_score,
    classification_mean_width_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
import pickle


def convert_form_anndata(adata, cell_annotation_col):

    adata.var_names_make_unique()
    exp_data = pd.DataFrame(
        data=adata.X.todense(), columns=adata.var_names, index=adata.obs.index
    ).to_numpy()
    labels = adata.obs[cell_annotation_col].to_numpy()
    gene_names = adata.var_names.to_numpy()

    return exp_data, labels, gene_names


sc_adata = sc.read_h5ad(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\STNav\data\processed\PipelineRun_2024_06_02-09_46_17_PM\scRNA\Files\preprocessed_adata.h5ad"
)
sc_adata.obs = sc_adata.obs[["dataset", "ann_level_3_transferred_label"]]
sc_adata.obs.rename(columns={"ann_level_3_transferred_label": "y_true"}, inplace=True)
# drop the rows that have y_ture as "Unknown"
sc_adata = sc_adata[sc_adata.obs.y_true != "Unknown"]
st_adata = sc.read_h5ad(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\STNav\data/processed/PipelineRun_2024_06_03-11_53_39_AM/ST/Files/deconvoluted_adata.h5ad"
)
# Intersect the genes of st_adata with the genes of sc_adata


import lightgbm as lgb
import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import optuna
from scipy.sparse import csr_matrix
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
from tqdm import tqdm


class SpatialPredictor:
    def __init__(self, sc_adata, st_adata):
        self.sc_adata = sc_adata
        self.st_adata = st_adata

    def feature_selector(self, sc_adata):
        sc_adata.X = sc_adata.layers["raw_counts"]
        exp_data, labels, gene_names = convert_form_anndata(sc_adata, "y_true")
        mags = sm.ScMags(data=exp_data, labels=labels, gene_ann=gene_names)
        sc_adata.X = sc_adata.layers["lognorm"]

        mags.filter_genes(nof_sel=1000)
        mags.sel_clust_marker(nof_markers=119)
        mrks_df = mags.get_markers()
        cell_type_markers_dict = {
            index.replace("C_", ""): [val for val in row.tolist()]
            for index, row in mrks_df.iterrows()
        }
        common_genes = list(
            set(
                [
                    val.upper()
                    for sublist in cell_type_markers_dict.values()
                    for val in sublist
                    if pd.notnull(val)
                ]
            )
        )
        return common_genes

    def train_model(self, test_size, metric, n_splits, n_trials, annotation, shuffle):
        cv_results_dict = {}
        training_plotting_metrics = {}
        model_name = "LGBM"

        self.st_adata.var_names = self.st_adata.var_names.str.upper()
        self.st_adata.var.index = self.st_adata.var.index.str.upper()

        self.sc_adata.var_names = self.sc_adata.var_names.str.upper()
        self.sc_adata.var.index = self.sc_adata.var.index.str.upper()

        self.sc_adata.var_names_make_unique()
        self.st_adata.var_names_make_unique()
        common_genes = self.sc_adata.var_names.intersection(self.st_adata.var_names)
        print(f"Number of common genes: {len(common_genes)}")

        print(f"Shape of sc_adata: {self.sc_adata.shape}")
        print(f"Shape of st_adata: {self.st_adata.shape}")
        self.sc_adata = self.sc_adata[:, common_genes]
        self.st_adata = self.st_adata[:, common_genes]
        print(f"Shape of sc_adata after subsetting: {self.sc_adata.shape}")
        print(f"Shape of st_adata after subsetting: {self.st_adata.shape}")

        data = pd.DataFrame(
            data=self.sc_adata.X.toarray(),
            index=self.sc_adata.obs.index,
            columns=self.sc_adata.var.index.str.upper(),
        )
        data[annotation] = self.sc_adata.obs[annotation]
        labels = data[annotation].cat.categories.to_numpy()

        selected_features = self.feature_selector(sc_adata=self.sc_adata)
        print(f"Selected features: {len(selected_features)}\n{selected_features}")

        X_complete = data.drop(annotation, axis=1)
        X = X_complete[selected_features]
        y = data[annotation]
        y_encoded, le = self.encode_labels(y, labels)

        X_train_cal, X_test, y_train_cal, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_cal,
            y_train_cal,
            test_size=0.2,
            random_state=42,
            stratify=y_train_cal,
        )

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=42)

        study = self.optimize_hyperparameters(
            X_train, y_train, cv, metric=metric, decoder=le, n_trials=n_trials
        )
        best_trial = study.best_trial
        best_params = best_trial.params
        best_cv_score = best_trial.value
        print(f"Best hyperparameters: {best_params}")
        print(f"Best score: {best_cv_score}")

        optimized_model = lgb.LGBMClassifier(**best_params)
        optimized_model.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(optimized_model, cv="prefit").fit(
            X_cal, y_cal
        )

        cv_results = cross_validate(
            optimized_model,
            X_train,
            y_train,
            cv=cv,
            scoring=metric,
            return_estimator=True,
            return_train_score=True,
            error_score="raise",
        )

        cv_results_dict["LGBM"] = cv_results
        pd.DataFrame.from_records(cv_results).to_excel(
            f"./cv_results.xlsx", index=False
        )

        cv_results_test = cv_results[f"test_score"]
        cv_results_train = cv_results[f"train_score"]
        current_model_score = cv_results_test.mean()

        training_plotting_metrics[model_name] = {
            f"test_{metric}": current_model_score,
            "std": cv_results_test.std(),
        }

        self.plot_training_curves(
            optimized_model,
            eval_score=metric,
            test_label_name="val",
            model_name=model_name,
            cv_results=cv_results,
            use_from_cross_val=False,
        )

        # y_pred_uncalibrated = optimized_model.predict(X_test)
        # y_pred_proba_uncalibrated = optimized_model.predict_proba(X_test)[:, 1]

        y_pred = calibrated.predict(X_test)
        y_pred_proba = calibrated.predict_proba(X_test)[:, 1]

        y_test_decoded = le.inverse_transform(y_test)
        y_pred_decoded = le.inverse_transform(y_pred)

        metrics_results = classification_report(
            y_pred=y_pred_decoded, y_true=y_test_decoded.tolist(), output_dict=True
        )
        self.save_results(metrics_results)

        sq.gr.spatial_neighbors(self.st_adata)
        self.st_adata.obsp["spatial_connectivities"] = csr_matrix(
            self.st_adata.obsp["spatial_connectivities"]
        )

        st_adata = self.st_adata[:, selected_features].copy()
        st_adata.X = st_adata.obsp["spatial_connectivities"].dot(st_adata.X)

        spatial_adata = pd.DataFrame.sparse.from_spmatrix(
            st_adata.X,
            index=st_adata.obs.index,
            columns=st_adata.var_names.str.upper(),
        )

        # Use just the optimized model to get the predictions
        predictions_non_conformal = self.predict_in_batches(calibrated, spatial_adata)
        predictions_non_conformal = le.inverse_transform(predictions_non_conformal)
        st_adata.obs["predicted_cell_type_non_conformal"] = pd.Categorical(
            predictions_non_conformal
        )

        # ## Use the conformal prediction
        # predictions_conformal = self.predict_with_conformal(
        #     calibrated, X_cal, y_cal, spatial_adata
        # )
        # predictions_conformal = le.inverse_transform(predictions_conformal)
        # st_adata.obs["predicted_cell_type_conformal"] = pd.Categorical(
        #     predictions_conformal
        # )

        # Save the optimized model
        with open("optimized_model.pkl", "wb") as f:
            pickle.dump(optimized_model, f)

        # Save the calibrated model
        with open("calibrated_model.pkl", "wb") as f:
            pickle.dump(calibrated, f)

        return (
            optimized_model,
            le,
            st_adata,
            predictions_non_conformal,
            # predictions_conformal,
        )

    def predict_with_conformal(self, model, X_cal, y_cal, data, batch_size=10000):
        conformal_predictions = []

        num_batches = int(np.ceil(data.shape[0] / batch_size))
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size

            batch = data[start:end]
            conformal_model, batch_prediction_sets, batch_conformal_sets = (
                self.apply_conformal_prediction(model, X_cal, y_cal, batch)
            )
            conformal_predictions.extend(batch_conformal_sets)
            # As the conformal_sets are not used, we can omit the third return value

        conformal_predictions = np.array(conformal_predictions)
        return conformal_predictions

    def predict_in_batches(self, model, data, batch_size=10000):
        predictions = []
        num_batches = int(np.ceil(data.shape[0] / batch_size))
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch = data[start:end]
            batch_predictions = model.predict(batch)
            predictions.append(batch_predictions)
        predictions = np.concatenate(predictions)
        return predictions

    def apply_conformal_prediction(self, model, X_cal, y_cal, X_test_external):
        mapie = MapieClassifier(model, method="aps", cv="prefit")
        mapie.fit(X_cal, y_cal)

        y_pred, y_ps = mapie.predict(
            X_test_external, alpha=0.05, include_last_label=True
        )
        y_ps_squeezed = [
            np.where(row)[0][0] if len(np.where(row)[0]) > 0 else -1
            for row in y_ps.squeeze()
        ]
        return mapie, y_pred, y_ps

    def encode_labels(self, y, cell_type):
        le = LabelEncoder()
        le.fit(y)
        y_encoded = le.transform(y)
        return y_encoded, le

    def optimize_hyperparameters(self, X, y, cv, decoder, metric, n_trials=25):
        def objective(trial):
            params = {
                "objective": "multiclass",
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "verbosity": -1,
            }

            model = lgb.LGBMClassifier(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, verbose=2)
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study

    def save_results(self, model_metrics, prefix: str = "test"):
        results = pd.json_normalize(model_metrics)
        results.columns = pd.MultiIndex.from_tuples(
            [tuple(col.split(".")) for col in results.columns]
        )
        results = results.T
        results.to_excel(f"./{prefix}_results.xlsx")

    def plot_training_curves(
        self,
        clf,
        test_label_name,
        eval_score,
        model_name,
        cv_results,
        use_from_cross_val=False,
    ):
        if hasattr(clf, "cv_results_"):
            val_scores = clf.cv_results_[f"mean_test_score"]
            train_scores = clf.cv_results_[f"mean_train_score"]
        else:
            val_scores = cv_results[f"test_score"]
            train_scores = cv_results[f"train_score"]

        plt.plot(val_scores, label=test_label_name)
        plt.plot(train_scores, label="train")
        plt.title(f"Validation Curve for {model_name} using {eval_score}.")
        plt.legend(loc="best")
        with plt.rc_context():
            plt.savefig(
                f"./TrainingCurve_{model_name}.png",
                bbox_inches="tight",
            )
        plt.close()

    def plot_results(self, cell_type, num_markers_list, metric_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(num_markers_list, metric_scores, label=self.metric, marker="o")
        plt.xlabel("Number of Markers")
        plt.ylabel(self.metric)
        plt.title(f"{self.metric} for {cell_type} on the test set")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{cell_type}_test_plot.png")
        plt.close()

    def save_best_results(self, dictio, filename="best_results.json"):
        with open(filename, "w") as f:
            json.dump(dictio, f)


# def main():
# Usage
evaluator = SpatialPredictor(sc_adata, st_adata)
(
    optimized_model,
    le,
    st_adata_pred,
    predictions_non_conformal,
) = evaluator.train_model(
    test_size=0.1,
    metric="balanced_accuracy",
    n_splits=7,
    n_trials=750,
    annotation="y_true",
    shuffle=True,
)
st_adata_pred.write_h5ad("predicted_adata_deconv.h5ad")

# TOOD: plot the scMAGS average reults, with the model results for each cell type and all cell types side by side as comparision
# if "__main__" == __name__:
#     main()

st_adata_pred.write_h5ad(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\STNav\notebooks\experimental\SpatialLGBM\predicted_adata_deconv.h5ad",
)
st_adata_pred.obs.columns
import matplotlib.pyplot as plt
import scanpy as sc

unique_cell_types = st_adata_pred.obs["predicted_cell_type_non_conformal"].unique()

for cell_type in unique_cell_types:
    # Create a subset of the data for the current cell type
    subset = st_adata_pred[
        st_adata_pred.obs["predicted_cell_type_non_conformal"] == cell_type
    ].copy()

    # Create a new figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the size as needed

    # Create the plot for the current cell type from the subset
    sc.pl.spatial(
        subset,
        color="predicted_cell_type_non_conformal",  # use the column from the subset
        size=0.9,
        alpha_img=0.5,
        library_id="Visium_HD_Human_Lung_Cancer",
        title=f"Spatial scatter for {cell_type} (subset)",
        ax=axs[0],  # plot on the first subplot
        show=False,  # do not show the plot yet
    )

    # Create the plot for the current cell type from the whole dataframe
    sc.pl.spatial(
        st_adata_pred,
        color=f"{cell_type}_Mean_LogNorm_Conn_Adj_scMAGS",  # use the column from the whole dataframe
        size=1.5,
        alpha_img=0.5,
        library_id="Visium_HD_Human_Lung_Cancer",
        title=f"Spatial scatter for {cell_type} (whole)",
        ax=axs[1],  # plot on the second subplot
        show=False,  # do not show the plot yet
    )

    # Save the figure
    plt.savefig(f"{cell_type}_scatter.png", dpi=1000)  # save the plot as a PNG file
    plt.show()
