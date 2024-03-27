import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from scipy.optimize import differential_evolution
from tqdm import tqdm
from sklearn.model_selection import KFold
from models.model_utils import roc_curve, auc, taylor_diagram, violin_plot, accuracy_score, precision_score, \
    recall_score, f1_score


class LogisticReg:
    def __init__(self, data_dict):
        self.x_train = data_dict["x_train"]
        self.y_train = data_dict["y_train"]

        self.x_test = data_dict["x_test"]
        self.y_test = data_dict["y_test"]

    def _build_model(self):
        self.model = LogisticRegression()

    def _fit_model(self):
        self.model.fit(self.x_train, self.y_train)

    def _predict_test(self):
        self.y_pred = self.model.predict(self.x_test)

    def _compute_visualize_metric(self):
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        taylor_diagram(self.y_test, self.y_pred)
        violin_plot(self.y_test, self.y_pred)

    def run(self):
        self._build_model()
        self._fit_model()
        self._predict_test()
        self._compute_visualize_metric()


class SupportVectorClassifier:
    def __init__(self, data_dict):
        self.x_train = data_dict["x_train"]
        self.y_train = data_dict["y_train"]

        self.x_test = data_dict["x_test"]
        self.y_test = data_dict["y_test"]

    def _build_model(self):
        self.model = SVC()

    def _fit_model(self):
        self.model.fit(self.x_train, self.y_train)

    def _predict_test(self):
        self.y_pred = self.model.predict(self.x_test)

    def _compute_visualize_metric(self):
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        taylor_diagram(self.y_test, self.y_pred)
        violin_plot(self.y_test, self.y_pred)

    def run(self):
        self._build_model()
        self._fit_model()
        self._predict_test()
        self._compute_visualize_metric()


class DTClassifier:
    def __init__(self, data_dict):
        self.x_train = data_dict["x_train"]
        self.y_train = data_dict["y_train"]

        self.x_test = data_dict["x_test"]
        self.y_test = data_dict["y_test"]

    def _build_model(self):
        self.model = DecisionTreeClassifier()

    def _fit_model(self):
        self.model.fit(self.x_train, self.y_train)

    def _predict_test(self):
        self.y_pred = self.model.predict(self.x_test)

    def _compute_visualize_metric(self):
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        taylor_diagram(self.y_test, self.y_pred)
        violin_plot(self.y_test, self.y_pred)

    def run(self):
        self._build_model()
        self._fit_model()
        self._predict_test()
        self._compute_visualize_metric()


class RFClassifier:
    def __init__(self, data_dict):
        self.x_train = data_dict["x_train"]
        self.y_train = data_dict["y_train"]

        self.x_test = data_dict["x_test"]
        self.y_test = data_dict["y_test"]

    def _build_model(self):
        self.model = RandomForestClassifier()

    def _fit_model(self):
        self.model.fit(self.x_train, self.y_train)

    def _predict_test(self):
        self.y_pred = self.model.predict(self.x_test)

    def _compute_visualize_metric(self):
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        taylor_diagram(self.y_test, self.y_pred)
        violin_plot(self.y_test, self.y_pred)

    def run(self):
        self._build_model()
        self._fit_model()
        self._predict_test()
        self._compute_visualize_metric()


class XGBClassifier:
    def __init__(self, data_dict):
        self.x_train = data_dict["x_train"]
        self.y_train = data_dict["y_train"]

        self.x_test = data_dict["x_test"]
        self.y_test = data_dict["y_test"]

    def _build_model(self):
        self.model = xgb.XGBClassifier()

    def _fit_model(self):
        self.model.fit(self.x_train, self.y_train)

    def _predict_test(self):
        self.y_pred = self.model.predict(self.x_test)

    def _compute_visualize_metric(self):
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        taylor_diagram(self.y_test, self.y_pred)
        violin_plot(self.y_test, self.y_pred)

    def run(self):
        self._build_model()
        self._fit_model()
        self._predict_test()
        self._compute_visualize_metric()


class LGBMClassifierCustom:
    def __init__(self, data_dict):
        self.x_train = data_dict["x_train"]
        self.y_train = data_dict["y_train"]

        self.x_test = data_dict["x_test"]
        self.y_test = data_dict["y_test"]

    def _build_model(self):
        self.model = lgb.LGBMClassifier(verbose=-1)

    def _fit_model(self):
        self.model.fit(self.x_train, self.y_train)

    def _predict_test(self):
        self.y_pred = self.model.predict(self.x_test)

    def _compute_visualize_metric(self):
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        taylor_diagram(self.y_test, self.y_pred)
        violin_plot(self.y_test, self.y_pred)

    def run(self):
        self._build_model()
        self._fit_model()
        self._predict_test()
        self._compute_visualize_metric()


class LGBMClassifierBO:
    def __init__(self, data_dict, pbounds, num_iterations):
        self.x_train = data_dict["x_train"]
        self.y_train = data_dict["y_train"]

        self.x_test = data_dict["x_test"]
        self.y_test = data_dict["y_test"]

        self.pbounds = pbounds
        self.num_iters = num_iterations
        self.params = None
        self.y_pred = None

    def _create_lgb_dataset(self):
        # Create LightGBM dataset
        self.dtrain = lgb.Dataset(self.x_train,
                                  label=self.y_train)
        self.dval = lgb.Dataset(self.x_test,
                                label=self.y_test)

    def _train_model(self):
        # Train the model with the given parameters
        self.model = lgb.train(self.params,
                               self.dtrain,
                               num_boost_round=1000,
                               valid_sets=[self.dval])
        # verbose_eval=False)

    def _evaluate_model(self):
        # Predict probabilities on validation set
        self.y_pred_proba = self.model.predict(self.x_test)

        # Predict classes
        self.y_pred = self.y_pred_proba.argmax(axis=1)

        # Calculate AUC
        self.auc_value = auc(self.y_test, self.y_pred)

    def _compute_visualize_metric(self):
        accuracy_val = accuracy_score(y_pred=self.y_pred, y_test=self.y_test)
        print("Accuracy of the Model: ", accuracy_val)

        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        taylor_diagram(self.y_test, self.y_pred)
        violin_plot(self.y_test, self.y_pred)

    def optimization_loop(self,
                          num_leaves,
                          max_depth,
                          learning_rate,
                          subsample,
                          colsample_bytree,
                          reg_alpha,
                          reg_lambda):
        self.params = {
            'num_leaves': int(num_leaves),
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': 'multiclass',
            'num_class': len(set(self.y_train)),  # Number of classes
            'metric': 'multi_logloss',  # Use multi_logloss for multiclass
            'verbosity': -1,
            'n_jobs': -1
        }

        self._create_lgb_dataset()
        self._train_model()
        self._evaluate_model()

        return self.auc_value

    def _define_optimizer(self):
        self.optimizer = BayesianOptimization(
            f=self.optimization_loop,
            pbounds=self.pbounds,
            random_state=42,
            verbose=2
        )

    def optimize(self):
        # Define the Optimizer
        self._define_optimizer()

        # Perform Bayesian Optimization
        self.optimizer.maximize(init_points=5, n_iter=self.num_iters)

    def _get_best_params(self):
        return self.optimizer.max["params"]

    def run(self):
        self.optimize()
        self.params = self._get_best_params()
        self.params["num_leaves"] = int(self.params["num_leaves"])
        self.params["max_depth"] = int(self.params["max_depth"])
        self.params["verbose"] = -1

        self._create_lgb_dataset()
        self._train_model()

        # Predict probabilities on validation set
        self.y_pred = self.model.predict(self.x_test)

        # self._evaluate_model()

        self._compute_visualize_metric()


class DELGBMModified:
    def __init__(self,
                 data_dict,
                 pbounds,
                 num_iterations,
                 num_cv_iterations,
                 tolerance,
                 random_state,
                 P):
        self.x = data_dict['x']
        self.y = data_dict['y']
        self.pbounds = pbounds
        self.num_iters = num_iterations
        self.num_cv_iterations = num_cv_iterations
        self.tol = tolerance
        self.seed = random_state
        self.P = P
        self.best_accuracy = 0
        self.params = None
        self.best_train_indices = None
        self.best_test_indices = None
        self.param_bounds = None
        self.optimizer = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def _train_model(self):
        # Train the model with the given parameters
        num_leaves = self.params["num_leaves"]
        max_depth = self.params["max_depth"]
        learning_rate = self.params["learning_rate"]
        # subsample = self.params["subsample"]
        # colsample_bytree = self.params["colsample_bytree"]
        # reg_alpha = self.params["reg_alpha"]
        # reg_lambda = self.params["reg_lambda"]

        # Create LightGBM classifier with given hyperparameters
        self.model = LGBMClassifier(learning_rate=learning_rate,
                                    num_leaves=int(num_leaves),
                                    max_depth=int(max_depth),
                                    # subsample=subsample,
                                    # colsample_bytree=colsample_bytree,
                                    # reg_alpha=reg_alpha,
                                    # reg_lambda=reg_lambda,
                                    verbose=-1)

        # Train the model
        self.model.fit(self.x_train, self.y_train)

    def _evaluate_model(self):
        # Predictions
        self.y_pred = self.model.predict(self.x_test)

        # Calculate AUC
        self.auc_value = auc(self.y_test, self.y_pred)

        # Calculate Accuracy
        self.accuracy = accuracy_score(y_pred=self.y_pred, y_test=self.y_test)

    def _compute_visualize_metric(self):
        accuracy_val = accuracy_score(y_pred=self.y_pred, y_test=self.y_test)
        print("Accuracy of the Model: ", accuracy_val)

        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        taylor_diagram(self.y_test, self.y_pred)
        violin_plot(self.y_test, self.y_pred)

    def objective_func(self, params):
        # num_leaves, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda = params
        num_leaves, max_depth, learning_rate = params

        # Create LightGBM classifier with given hyperparameters
        lgbm_model = LGBMClassifier(learning_rate=learning_rate,
                                    num_leaves=int(num_leaves),
                                    max_depth=int(max_depth),
                                    # subsample=subsample,
                                    # colsample_bytree=colsample_bytree,
                                    # reg_alpha=reg_alpha,
                                    # reg_lambda=reg_lambda,
                                    verbose=-1)

        # Train the model
        lgbm_model.fit(self.x_train, self.y_train)

        # Predictions
        y_pred = lgbm_model.predict(self.x_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test=self.y_test, y_pred=y_pred)

        # Minimize negative accuracy
        return -accuracy

    def optimize(self):
        # Define Parameter List:
        self.param_bounds = [[param[0], param[1]] for param in self.pbounds.values()]

        # Define Optimizer and Optimize:
        self.optimizer = differential_evolution(
            func=self.objective_func,
            bounds=self.param_bounds,
            maxiter=self.num_iters,
            tol=self.tol,
            seed=self.seed
        )

    def _get_best_params(self):
        # Extract best hyperparameters
        best_params = self.optimizer.x
        # num_leaves, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda = best_params
        num_leaves, max_depth, learning_rate = best_params

        num_leaves = int(num_leaves)
        max_depth = int(max_depth)

        # Define Parameter Dictionary
        best_params_dict = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            # "subsample": subsample,
            # "colsample_bytree": colsample_bytree,
            # "reg_alpha": reg_alpha,
            # "reg_lambda": reg_lambda
        }
        return best_params_dict

    def _compute_lpo_indices(self):
        start_indices = np.arange(0, self.num_samples, self.P)
        self.indices_list = [list(range(start_idx, start_idx + self.P)) for start_idx in start_indices]

    def _run_cv(self):
        self.num_classes = len(np.unique(self.y))
        self.num_samples, self.num_features = self.x.shape

        # Shuffle the data
        self.indices = np.random.permutation(self.num_samples)
        self.x_shuffled = self.x[self.indices]
        self.y_shuffled = self.y[self.indices]

        # Compute Indices for Leave P Out Cross Validation
        self._compute_lpo_indices()

        # Initialize list to store accuracy scores
        self.accuracy_scores_list = []

        # Perform Leave P Out Cross Validation
        i = 0

        for test_indices in tqdm(self.indices_list):
            # print("Set: ", i)
            # Use remaining indices for training
            train_indices = np.delete(np.arange(self.num_samples),
                                      test_indices)

            # Creating Training and Testing Set
            self.x_train, self.y_train = self.x_shuffled[train_indices], self.y_shuffled[train_indices]
            self.x_test, self.y_test = self.x_shuffled[test_indices], self.y_shuffled[test_indices]

            # Start Hyperparameter Optimization
            self.optimize()
            self.params = self._get_best_params()
            self.params["verbose"] = -1

            # Estimate Accuracy Metric
            self._train_model()
            self._evaluate_model()

            if self.accuracy > self.best_accuracy:
                self.best_train_indices = train_indices
                self.best_test_indices = test_indices
                self.best_params = self.params

            self.accuracy_scores_list.append(self.accuracy)
            i += 1

        print("Mean Accuracy Scores: ", np.mean(self.accuracy_scores_list))

    def run(self):
        # Run Cross Validation Loop
        self._run_cv()

        # Run the model for best train set and best parameters
        self.x_train, self.y_train = self.x_shuffled[self.best_train_indices], self.y_shuffled[self.best_train_indices]
        self.x_test, self.y_test = self.x_shuffled[self.best_test_indices], self.y_shuffled[self.best_test_indices]
        self.params = self.best_params

        self._train_model()
        self._evaluate_model()

        self._compute_visualize_metric()


class PBLGBM:
    def __init__(self,
                 data_dict,
                 pbounds,
                 num_iterations,
                 num_configs,
                 num_inner_splits=10,
                 num_outer_splits=10,
                 random_state=8420,
                 exploration_rate=0.1):
        self.x = data_dict['x']
        self.y = data_dict['y']
        self.pbounds = pbounds
        self.num_iters = num_iterations
        self.num_configs = num_configs
        self.num_inner_splits = num_inner_splits
        self.num_outer_splits = num_outer_splits
        self.seed = random_state
        self.exploration_rate = exploration_rate
        self.best_accuracy = 0
        self.population = []
        self.scores = []
        self.params = None
        self.sorted_indices = None
        self.best_train_indices = None
        self.best_test_indices = None
        self.param_bounds = None
        self.optimizer = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def _train_model(self):
        # Train the model with the given parameters
        num_leaves = self.params["num_leaves"]
        max_depth = self.params["max_depth"]
        learning_rate = self.params["learning_rate"]
        # subsample = self.params["subsample"]
        # colsample_bytree = self.params["colsample_bytree"]
        # reg_alpha = self.params["reg_alpha"]
        # reg_lambda = self.params["reg_lambda"]

        # Create LightGBM classifier with given hyperparameters
        self.model = LGBMClassifier(learning_rate=learning_rate,
                                    num_leaves=int(num_leaves),
                                    max_depth=int(max_depth),
                                    # subsample=subsample,
                                    # colsample_bytree=colsample_bytree,
                                    # reg_alpha=reg_alpha,
                                    # reg_lambda=reg_lambda,
                                    verbose=-1)

        # Train the model
        self.model.fit(self.x_train, self.y_train)

    def _evaluate_model(self):
        # Predictions
        self.y_pred = self.model.predict(self.x_test)

        # Calculate AUC
        self.auc_value = auc(self.y_test, self.y_pred)

        # Calculate Accuracy
        self.accuracy = accuracy_score(y_pred=self.y_pred, y_test=self.y_test)

    def _compute_visualize_metric(self):
        accuracy_val = accuracy_score(y_pred=self.y_pred, y_test=self.y_test)
        print("Accuracy of the Model: ", accuracy_val)

        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        taylor_diagram(self.y_test, self.y_pred)
        violin_plot(self.y_test, self.y_pred)

    # Create a random configuration
    def create_random_config(self):
        return {param: np.random.uniform(bounds[0], bounds[1]) for param, bounds in self.pbounds.items()}

    # Perturb a configuration
    def perturb(self, config):
        param = np.random.choice(list(self.pbounds.keys()))
        new_config = config.copy()
        new_config[param] = np.random.choice(self.pbounds[param])
        return new_config

    # Exploit between two configurations
    def exploit(self, target_config, source_config):
        param = np.random.choice(list(self.pbounds.keys()))
        new_config = target_config.copy()
        new_config[param] = source_config[param]
        return new_config

    # Population Based Training (PBT)
    def pbt(self):
        # Initialize population
        self.population = []
        self.scores = []
        for _ in range(self.num_configs):
            self.population.append(self.create_random_config())

        for self.params in self.population:
            self._train_model()
            self._evaluate_model()
            self.scores.append(self.accuracy)

        # Main loop
        for iteration in tqdm(range(self.num_iters)):
            # print(f"Iteration {iteration + 1}/{self.num_iters}")

            # Sort population by score
            self.sorted_indices = np.argsort(self.scores)

            self.population = [self.population[i] for i in self.sorted_indices]
            self.scores = [self.scores[i] for i in self.sorted_indices]

            # Perform exploration
            num_explored = int(self.exploration_rate * self.num_configs)

            for i in range(num_explored):
                self.params = self.perturb(self.population[i])
                self._train_model()
                self._evaluate_model()

                if self.accuracy > self.scores[-1]:
                    self.population[-1] = self.params
                    self.scores[-1] = self.accuracy

            # Perform exploitation
            num_exploited = self.num_configs - num_explored
            for i in range(num_exploited):
                target_index = np.random.choice(num_explored)
                source_index = np.random.choice(num_explored)
                self.params = self.exploit(self.population[target_index],
                                           self.population[source_index])
                self._train_model()
                self._evaluate_model()

                if self.accuracy > self.scores[-1]:
                    self.population[-1] = self.params
                    self.scores[-1] = self.accuracy

        # Return the best configuration
        best_config = self.population[np.argmax(self.scores)]
        return best_config

    def _run_cv(self):
        self.outer_cv = KFold(n_splits=self.num_outer_splits,
                              shuffle=True,
                              random_state=self.seed)
        for train_index, test_index in self.outer_cv.split(self.x):
            self.x_train_outer, self.x_test_outer = self.x[train_index], self.x[test_index]
            self.y_train_outer, self.y_test_outer = self.y[train_index], self.y[test_index]

            # Inner cross-validation loop for hyperparameter tuning
            self.inner_cv = KFold(n_splits=self.num_inner_splits, shuffle=True, random_state=42)
            for train_index_inner, test_index_inner in self.inner_cv.split(self.x_train_outer):
                self.x_train_inner, self.x_test_inner = self.x_train_outer[train_index_inner], \
                                                        self.x_train_outer[test_index_inner]
                self.y_train_inner, self.y_test_inner = self.y_train_outer[train_index_inner], \
                                                        self.y_train_outer[test_index_inner]

                # Define Training and Testing Set and train the model
                self.x_train = self.x_train_inner
                self.y_train = self.y_train_inner

                self.x_test = self.x_test_inner
                self.y_test = self.y_test_inner

                self.params = self.pbt()

                # Evaluate Model for best params:
                self._train_model()
                self._evaluate_model()

                # Check if it is better than previous best
                if self.accuracy > self.best_accuracy:
                    self.best_accuracy = self.accuracy
                    self.best_train_indices = train_index
                    self.best_test_indices = test_index
                    self.best_params = self.params

            # Train Model with best params for inner loop
            self.x_train = self.x_train_outer
            self.x_test = self.x_test_outer

            self.y_train = self.y_train_outer
            self.y_test = self.y_test_outer

            self.params = self.best_params

            self._train_model()
            self._evaluate_model()

            self.scores.append(self.accuracy)

        mean_accuracy = np.mean(self.scores)
        print("Mean Accuracy: ", mean_accuracy)

    def run(self):
        # Run Cross Validation Loop
        self._run_cv()

        # Run the model for best train set and best parameters
        self.x_train, self.y_train = self.x[self.best_train_indices], self.y[self.best_train_indices]
        self.x_test, self.y_test = self.x[self.best_test_indices], self.y[self.best_test_indices]
        self.params = self.best_params

        self._train_model()
        self._evaluate_model()

        self._compute_visualize_metric()
