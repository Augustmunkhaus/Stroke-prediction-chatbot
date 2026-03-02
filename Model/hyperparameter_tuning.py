# """
# The following method is for hyperparameter tuning and is moved to a separate file.
# """
# def train_models(self):
#     """Train different ML models with hyperparameter tuning using accuracy"""
#     print("\n=== MODEL TRAINING WITH HYPERPARAMETER TUNING (ACCURACY) ===")
#
#     # Use accuracy as scoring metric
#     scoring = 'accuracy'
#
#     # Random Forest with GridSearchCV
#     print("Tuning Random Forest...")
#     rf_param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [5, 10, 15, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'class_weight': ['balanced', 'balanced_subsample']
#     }
#
#     rf_base = RandomForestClassifier(random_state=42)
#     rf_grid = GridSearchCV(
#         rf_base, rf_param_grid,
#         cv=5, scoring=scoring,
#         n_jobs=-1, verbose=1
#     )
#     rf_grid.fit(self.X_train, self.y_train)
#     self.models['Random Forest'] = rf_grid.best_estimator_
#     self.best_params['Random Forest'] = rf_grid.best_params_
#     print(f"Best RF params: {rf_grid.best_params_}")
#     print(f"Best RF CV accuracy: {rf_grid.best_score_:.4f}")
#
#     # Logistic Regression with GridSearchCV
#     print("\nTuning Logistic Regression...")
#     lr_param_grid = {
#         'C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'penalty': ['l1', 'l2', 'elasticnet'],
#         'solver': ['liblinear', 'saga'],
#         'class_weight': ['balanced', None],
#         'l1_ratio': [0.1, 0.5, 0.7, 0.9]  # Only used with elasticnet
#     }
#
#     lr_base = LogisticRegression(random_state=42, max_iter=2000)
#     lr_grid = GridSearchCV(
#         lr_base, lr_param_grid,
#         cv=5, scoring=scoring,
#         n_jobs=-1, verbose=1
#     )
#     lr_grid.fit(self.X_train_scaled, self.y_train)
#     self.models['Logistic Regression'] = lr_grid.best_estimator_
#     self.best_params['Logistic Regression'] = lr_grid.best_params_
#     print(f"Best LR params: {lr_grid.best_params_}")
#     print(f"Best LR CV accuracy: {lr_grid.best_score_:.4f}")
#
#     # Neural Network with RandomizedSearchCV (faster for many parameters)
#     print("\nTuning Neural Network...")
#     nn_param_dist = {
#         'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25), (200, 100), (150, 75, 25)],
#         'learning_rate_init': uniform(0.0001, 0.01),
#         'alpha': uniform(0.0001, 0.1),
#         'batch_size': ['auto', 32, 64, 128],
#         'max_iter': [500, 1000, 2000],
#         'early_stopping': [True],
#         'validation_fraction': [0.1, 0.2],
#         'solver': ['adam', 'lbfgs']
#     }
#
#     # Calculate class weights for sample weighting
#     classes = np.unique(self.y_train)
#     class_weights = compute_class_weight('balanced', classes=classes, y=self.y_train)
#     class_weight_dict = dict(zip(classes, class_weights))
#     sample_weights = np.array([class_weight_dict[y] for y in self.y_train])
#
#     nn_base = MLPClassifier(random_state=42)
#
#     # Manual search with sample weights and accuracy scoring
#     print("Training Neural Network with sample weights...")
#     best_score = 0
#     best_params = None
#     best_model = None
#
#     np.random.seed(42)
#     for i in range(20):  # Try 20 combinations
#         params = {}
#         for param, dist in nn_param_dist.items():
#             if hasattr(dist, 'rvs'):  # For scipy distributions
#                 params[param] = dist.rvs()
#             else:  # For lists
#                 params[param] = random.choice(dist)
#
#         try:
#             nn_temp = MLPClassifier(**params, random_state=42)
#             # Cross-validate with sample weights using accuracy
#             scores = []
#             from sklearn.model_selection import StratifiedKFold
#             skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#
#             for train_idx, val_idx in skf.split(self.X_train_scaled, self.y_train):
#                 X_tr, X_val = self.X_train_scaled[train_idx], self.X_train_scaled[val_idx]
#                 y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
#                 sw_tr = sample_weights[train_idx]
#
#                 nn_temp.fit(X_tr, y_tr, sample_weight=sw_tr)
#                 y_pred = nn_temp.predict(X_val)
#                 score = accuracy_score(y_val, y_pred)
#                 scores.append(score)
#
#             avg_score = np.mean(scores)
#             if avg_score > best_score:
#                 best_score = avg_score
#                 best_params = params
#                 best_model = MLPClassifier(**params, random_state=42)
#
#         except Exception as e:
#             continue  # Skip problematic parameter combinations
#
#     # Train final model with best parameters
#     if best_model is not None:
#         best_model.fit(self.X_train_scaled, self.y_train, sample_weight=sample_weights)
#         self.models['Neural Network'] = best_model
#         self.best_params['Neural Network'] = best_params
#         print(f"Best NN params: {best_params}")
#         print(f"Best NN CV accuracy: {best_score:.4f}")
#     else:
#         # Fallback to simple model
#         fallback_params = {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'early_stopping': True}
#         nn_fallback = MLPClassifier(**fallback_params, random_state=42)
#         nn_fallback.fit(self.X_train_scaled, self.y_train, sample_weight=sample_weights)
#         self.models['Neural Network'] = nn_fallback
#         self.best_params['Neural Network'] = fallback_params
#         print("Using fallback NN parameters")
#
#     # Save best parameters to file
#     self.save_best_params()
#     print("\n=== HYPERPARAMETER TUNING COMPLETE ===")

# """
# The following method saves the results of the tuning process.
# """
# def save_best_params(self):
#     """Save best parameters to JSON file"""
#     # Convert numpy types to native Python types for JSON serialization
#     params_to_save = {}
#     for model_name, params in self.best_params.items():
#         params_to_save[model_name] = {}
#         for key, value in params.items():
#             if isinstance(value, np.integer):
#                 params_to_save[model_name][key] = int(value)
#             elif isinstance(value, np.floating):
#                 params_to_save[model_name][key] = float(value)
#             else:
#                 params_to_save[model_name][key] = value
#
#     with open(self.best_params_file, 'w') as f:
#         json.dump(params_to_save, f, indent=2)
#     print(f"Best parameters saved to {self.best_params_file}")

# """
# The following method is a utility to print the results of the tuning process.
# """
# def print_best_params(self):
#     """Print the best parameters found"""
#     if self.load_best_params():
#         print("\n=== BEST PARAMETERS FOUND ===")
#         for model_name, params in self.best_params.items():
#             print(f"\n{model_name}:")
#             for param, value in params.items():
#                 print(f"  {param}: {value}")
#     else:
#         print("No saved parameters found. Run hyperparameter tuning first.")

# """
# The following function is a utility to show the results of the tuning process.
# """
# def show_best_params():
#     """Utility function to show saved best parameters"""
#     stroke_model = StrokePredictionModel('dummy.csv')  # Just to access the method
#     stroke_model.print_best_params()

