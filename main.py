import os
import json
import numpy as np
# import wandb
import argparse
from sklearn.model_selection import train_test_split
from data.data import VietnamDataset, IndianDataset
from models.models import LogisticReg, \
    SupportVectorClassifier, DTClassifier, RFClassifier, XGBClassifier, \
    LGBMClassifierCustom, LGBMClassifierBO, DELGBMModified, PBLGBM
from models.model_utils import do_mann_whiteney_test, do_pc_columns_analysis, \
    do_clustering_analysis, visualize_mannwhiteney_results

RANDOM_STATE = 8420


def create_save_folders(output_path):
    base_dir = output_path["base_dir"]
    date_dir = output_path["date_dir"]
    run_dir = output_path["run_dir"]
    plots_dir = output_path["plots_dir"]
    model_dir = output_path["model_dir"]
    checkpoints_dir = output_path["checkpoints_dir"]

    run_path = os.path.join(base_dir, date_dir, run_dir)
    plots_path = os.path.join(base_dir, date_dir, run_dir, plots_dir)
    model_path = os.path.join(base_dir, date_dir, run_dir, model_dir)
    checkpoints_path = os.path.join(base_dir, date_dir, run_dir, checkpoints_dir)

    if os.path.exists(run_path):
        pass
    else:
        os.makedirs(run_path)

    if os.path.exists(plots_path):
        pass
    else:
        os.makedirs(plots_path)

    if os.path.exists(model_path):
        pass
    else:
        os.makedirs(model_path)

    if os.path.exists(checkpoints_path):
        pass
    else:
        os.makedirs(checkpoints_path)


def get_save_paths(output_path):
    base_dir = output_path["base_dir"]
    date_dir = output_path["date_dir"]
    run_dir = output_path["run_dir"]
    plots_dir = output_path["plots_dir"]
    model_dir = output_path["model_dir"]
    checkpoints_dir = output_path["checkpoints_dir"]

    plots_path = os.path.join(base_dir, date_dir, run_dir, plots_dir)
    model_path = os.path.join(base_dir, date_dir, run_dir, model_dir)
    checkpoints_path = os.path.join(base_dir, date_dir, run_dir, checkpoints_dir)
    results_arr_path = os.path.join(base_dir, date_dir, run_dir, "results_array.npz")

    paths_dir = {
        "plots_path": plots_path,
        "model_path": model_path,
        "checkpoints_path": checkpoints_path,
        "result_array_save_path": results_arr_path
    }
    return paths_dir


def run(args):
    runner_config = args.runner_configs_path
    f_runner = open(runner_config, "r")
    runner = json.load(f_runner)

    do_save = args.do_save

    # wandb.init()

    input_path_arg = runner["inputs_json_path"]
    config_path_arg = runner["config_json_path"]
    output_path_arg = runner["outputs_json_path"]
    data_vietnam_path_arg = runner["data_vietnam_json_path"]
    data_indian_path_arg = runner["data_indian_json_path"]

    f_input = open(input_path_arg, "r")
    input_config = json.load(f_input)

    f_output = open(output_path_arg, "r")
    output_config = json.load(f_output)

    f_config = open(config_path_arg, "r")
    config = json.load(f_config)

    f_data_vietnam = open(data_vietnam_path_arg, "r")
    data_vietnam_config = json.load(f_data_vietnam)

    f_data_indian = open(data_indian_path_arg, "r")
    data_indian_config = json.load(f_data_indian)

    create_save_folders(output_path=output_config)

    save_paths = get_save_paths(output_path=output_config)

    if input_config["process_data_from_scratch"]:
        vietnam_data = VietnamDataset(paths=data_vietnam_config)
        print("-" * 75)
        print("Data Preparation for Vietnam Dataset")
        print("-" * 75)
        vietnam_data.run()
        print()

        indian_data = IndianDataset(paths=data_indian_config)
        print("-" * 75)
        print("Data Preparation for Indian Dataset")
        print("-" * 75)
        indian_data.run()

    if input_config["use_general_model"]:
        vietnam_path = input_config["vietnam_path"]
        # Load arrays from the .npz file
        vietnam_data = np.load(vietnam_path)

        # Access individual arrays
        x_vietnam = vietnam_data['x']
        y_classification_vietnam = vietnam_data['y_classification']
        y_regression_vietnam = vietnam_data['y_regression']
        columns_vietnam = vietnam_data['columns']

        x = x_vietnam
        y = y_classification_vietnam
        y = y.reshape(-1, )

        # Splitting the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE,
                                                            shuffle=True)

        data_dict = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }

        # Logistic Regression Model
        logistic_regression = LogisticReg(data_dict=data_dict)
        logistic_regression.run()

        # Support Vector Model
        support_vector = SupportVectorClassifier(data_dict=data_dict)
        support_vector.run()

        # Decision Tree Model
        decision_tree = DTClassifier(data_dict=data_dict)
        decision_tree.run()

        # Random Forest Model
        random_forest = RFClassifier(data_dict=data_dict)
        random_forest.run()

        # XGBoost Model
        xg_boost = XGBClassifier(data_dict=data_dict)
        xg_boost.run()

        # LightGBM Model
        lgbm = LGBMClassifierCustom(data_dict=data_dict)
        lgbm.run()

    if input_config["use_bo_lgbm"]:
        vietnam_path = input_config["vietnam_path"]
        # Load arrays from the .npz file
        vietnam_data = np.load(vietnam_path)

        # Access individual arrays
        x_vietnam = vietnam_data['x']
        y_classification_vietnam = vietnam_data['y_classification']
        y_regression_vietnam = vietnam_data['y_regression']
        columns_vietnam = vietnam_data['columns']

        x = x_vietnam
        y = y_classification_vietnam
        y = y.reshape(-1, )

        # Splitting the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE,
                                                            shuffle=True)

        data_dict = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }

        # LightGBM + Bayesian Optimization Model

        # Define parameter bounds for optimization
        pbounds = {
            'num_leaves': (20, 100),
            'max_depth': (5, 15),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.5, 1),
            'reg_alpha': (0, 1),
            'reg_lambda': (0, 1)
        }

        lgbm_bo = LGBMClassifierBO(data_dict=data_dict, pbounds=pbounds, num_iterations=10)
        lgbm_bo.run()

    if input_config["use_de_lgbm"]:
        vietnam_path = input_config["vietnam_path"]
        # Load arrays from the .npz file
        vietnam_data = np.load(vietnam_path)

        # Access individual arrays
        x_vietnam = vietnam_data['x']
        y_classification_vietnam = vietnam_data['y_classification']
        y_regression_vietnam = vietnam_data['y_regression']
        columns_vietnam = vietnam_data['columns']

        x = x_vietnam
        y = y_classification_vietnam
        y = y.reshape(-1, )

        # Novel LightGBM Model
        data_dict_new = {
            'x': x,
            'y': y
        }

        pbounds = {
            'num_leaves': (20, 100),
            'max_depth': (5, 15),
            'learning_rate': (0.01, 0.3),
            # 'subsample': (0.5, 1),
            # 'colsample_bytree': (0.5, 1),
            # 'reg_alpha': (0, 1),
            # 'reg_lambda': (0, 1)
        }

        lgbm_ed = DELGBMModified(data_dict=data_dict_new,
                                 pbounds=pbounds,
                                 num_iterations=10,
                                 num_cv_iterations=10,
                                 tolerance=1e-3,
                                 random_state=8420,
                                 P=10)
        lgbm_ed.run()

    if input_config["use_pb_lgbm"]:
        vietnam_path = input_config["vietnam_path"]
        # Load arrays from the .npz file
        vietnam_data = np.load(vietnam_path)

        # Access individual arrays
        x_vietnam = vietnam_data['x']
        y_classification_vietnam = vietnam_data['y_classification']
        y_regression_vietnam = vietnam_data['y_regression']
        columns_vietnam = vietnam_data['columns']

        x = x_vietnam
        y = y_classification_vietnam
        y = y.reshape(-1, )

        # Novel LightGBM Model
        data_dict_new = {
            'x': x,
            'y': y
        }

        pbounds = {
            'num_leaves': (20, 100),
            'max_depth': (5, 15),
            'learning_rate': (0.01, 0.3),
            # 'subsample': (0.5, 1),
            # 'colsample_bytree': (0.5, 1),
            # 'reg_alpha': (0, 1),
            # 'reg_lambda': (0, 1)
        }

        pblgbm_ed = PBLGBM(data_dict=data_dict_new,
                           pbounds=pbounds,
                           num_iterations=10,
                           num_configs=10,
                           num_inner_splits=3,
                           num_outer_splits=5)
        pblgbm_ed.run()

    if input_config["explain_results"]:
        # TODO: SHAP Implementation
        # TODO: Feature Importance Analysis
        # TODO: Explanation of some examples
        pass

    if input_config["do_mann_whiteney_test"]:
        vietnam_path = input_config["vietnam_path"]

        # Load arrays from the .npz file
        vietnam_data = np.load(vietnam_path)

        # Access individual arrays
        x_vietnam = vietnam_data['x']
        y_classification_vietnam = vietnam_data['y_classification']
        columns = vietnam_data["columns"].tolist()

        # TODO: Mann Whiteney Test
        mannwhiteney_results_df = do_mann_whiteney_test(x=x_vietnam,
                                                        y=y_classification_vietnam,
                                                        columns=columns)

        print()
        visualize_mannwhiteney_results(mannwhiteney_results_df,
                                       save_paths=save_paths)

    if input_config["do_geochemical_component_analysis"]:
        vietnam_path = input_config["vietnam_path"]

        # Load arrays from the .npz file
        vietnam_data = np.load(vietnam_path)

        # Access individual arrays
        x_vietnam = vietnam_data['x']
        y_classification_vietnam = vietnam_data['y_classification']
        columns = vietnam_data["columns"].tolist()

        # TODO: Principal Component Analysis of each feature
        do_pc_columns_analysis(x=x_vietnam,
                               y=y_classification_vietnam,
                               columns=columns,
                               save_paths=save_paths)

    if input_config["do_clustering_analysis"]:
        vietnam_path = input_config["vietnam_path"]

        # Load arrays from the .npz file
        vietnam_data = np.load(vietnam_path)

        # Access individual arrays
        x_vietnam = vietnam_data['x']
        y_classification_vietnam = vietnam_data['y_classification']

        # TODO: Clustering Analysis
        do_clustering_analysis(x=x_vietnam,
                               y=y_classification_vietnam,
                               save_paths=save_paths)


if __name__ == "__main__":
    print("Running the system..")
    parser = argparse.ArgumentParser(description='Running the system')
    parser.add_argument("--do_parse", type=int, required=True)
    parser.add_argument("--runner_configs_path", type=str)
    parser.add_argument("--do_save", type=int)
    arguments = parser.parse_args()
    if arguments.do_parse:
        run(arguments)
    else:
        print("Please run this system through the terminal with required configuration file")
