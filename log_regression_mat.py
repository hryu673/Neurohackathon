import os
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import scipy.io

class LogisticRegressionClassifier:  # Renamed class to match actual algorithm used
    def __init__(self, data_dir="./data", sr=10000, n_mfcc=13,
                 logreg_solver='lbfgs', logreg_max_iter=1000, logreg_C=1.0,
                 mat_var_name_class0='trials',
                 mat_var_name_class1='StimContract',
                 random_state=42
                ):
        self.model = LogisticRegression(
            solver=logreg_solver,
            max_iter=logreg_max_iter,
            C=logreg_C,
            random_state=random_state
        )
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.mat_var_name_class0 = mat_var_name_class0
        self.mat_var_name_class1 = mat_var_name_class1
        self.random_state = random_state

        # Initialize attributes to store the test split from training data
        self.X_test_split = np.array([])
        self.y_test_split = np.array([])

        if not self.mat_var_name_class0 or not self.mat_var_name_class1:
            print("CRITICAL ERROR: MATLAB variable names for class0 and class1 data must be provided.")
            self.model = None # Mark model as not usable
            return

        if data_dir:
            print(f"Initializing Logistic Regression model. Attempting to train with data from: {data_dir}")
            try:
                self.train(data_dir) # This will now also create the test split
            except Exception as e:
                print(f"Model training failed during __init__: {e}")
                self.model = None
                print("Model is not trained.")
        else:
            print("Warning: No data directory provided during initialization. Model is not trained.")

    def extract_features_from_single_channel(self, data_1d: np.ndarray) -> np.ndarray:
        if data_1d.size == 0:
            raise ValueError("Data segment is empty, cannot extract features.")
        data_1d = data_1d.astype(np.float32)
        mfccs = librosa.feature.mfcc(y=data_1d, sr=self.sr, n_mfcc=self.n_mfcc)
        mean_mfccs = np.mean(mfccs, axis=1)
        std_mfccs = np.std(mfccs, axis=1)
        features = np.concatenate([mean_mfccs, std_mfccs])
        return features

    def load_dataset(self, root_dir: str) -> tuple[np.ndarray, np.ndarray]:
        X_all_datapoints, y_labels_for_datapoints = [], []
        # Track which file and trial each sample comes from to avoid data leakage
        sample_metadata = []
        
        print(f"Loading all dataset trials from {root_dir}...")
        for label_dir_name in sorted(os.listdir(root_dir)):
            full_dir_path = os.path.join(root_dir, label_dir_name)
            if not os.path.isdir(full_dir_path):
                continue

            current_label = -1
            mat_variable_name_for_current_class = None
            if "relax" in label_dir_name.lower() or "class0" in label_dir_name.lower():
                current_label = 0
                mat_variable_name_for_current_class = self.mat_var_name_class0
            elif "contract" in label_dir_name.lower() or "class1" in label_dir_name.lower():
                current_label = 1
                mat_variable_name_for_current_class = self.mat_var_name_class1
            else:
                print(f"Warning: Dir '{label_dir_name}' not recognized as class dir. Skipped.")
                continue
            
            print(f"Processing dir: {label_dir_name}, label: {current_label}, expecting .mat var: '{mat_variable_name_for_current_class}'")

            mat_files_in_dir = [f for f in os.listdir(full_dir_path) if f.lower().endswith(".mat")]
            if not mat_files_in_dir:
                print(f"Warning: No .mat files in {full_dir_path}. Skipping.")
                continue
            
            for fname in mat_files_in_dir:
                fpath = os.path.join(full_dir_path, fname)
                file_id = f"{label_dir_name}/{fname}"  # Unique file identifier
                print(f"  Loading from .mat file: {fname}")
                try:
                    mat_contents = scipy.io.loadmat(fpath)
                    if mat_variable_name_for_current_class not in mat_contents:
                        raise ValueError(f"Var '{mat_variable_name_for_current_class}' not found in {fname}.")
                    data_cell_array = mat_contents[mat_variable_name_for_current_class]
                    if not (isinstance(data_cell_array, np.ndarray) and data_cell_array.dtype == 'object' and
                            data_cell_array.ndim == 2 and data_cell_array.shape[1] == 1):
                        raise ValueError(f"Var '{mat_variable_name_for_current_class}' in {fname} not Nx1 MATLAB cell array.")
                    num_trials = data_cell_array.shape[0]
                    if num_trials == 0:
                        print(f"Warning: Cell array '{mat_variable_name_for_current_class}' in {fname} empty. Skipping.")
                        continue
                    print(f"    Found {num_trials} trials in {fname} using var '{mat_variable_name_for_current_class}'.")
                    for i in range(num_trials):
                        try:
                            trial_data_1d = data_cell_array[i, 0].squeeze()
                            if trial_data_1d.ndim != 1:
                                raise ValueError(f"Trial {i+1} data not 1D. Shape: {data_cell_array[i,0].shape}")
                            trial_features = self.extract_features_from_single_channel(trial_data_1d)
                            if trial_features.size > 0:
                                X_all_datapoints.append(trial_features)
                                y_labels_for_datapoints.append(current_label)
                                # Save metadata for this sample
                                sample_metadata.append({
                                    'file': file_id,
                                    'trial': i,
                                    'class': current_label
                                })
                        except Exception as trial_e:
                            print(f"    Error processing trial {i+1} in {fname}: {trial_e}. Skipping trial.")
                except Exception as file_e:
                    print(f"Error processing file {fpath} (skipped): {file_e}")
        
        if not X_all_datapoints:
            print("CRITICAL: No data points extracted from any files.")
            return np.array([]), np.array([]), []
            
        return np.array(X_all_datapoints), np.array(y_labels_for_datapoints), sample_metadata

    def perform_fixed_split(self, X_all, y_all, sample_metadata, samples_per_class=20):
        """Perform a split with exactly N samples per class for training, while avoiding data leakage."""
        
        print(f"\nPerforming fixed split with {samples_per_class} samples per class for training...")
        
        # Group samples by file and class to prevent data leakage
        files_by_class = {0: {}, 1: {}}
        
        # Assign each sample to its source file and class
        for i, meta in enumerate(sample_metadata):
            cls = meta['class']
            file_id = meta['file']
            
            if file_id not in files_by_class[cls]:
                files_by_class[cls][file_id] = []
            
            files_by_class[cls][file_id].append(i)
        
        # Shuffle files within each class
        np.random.seed(self.random_state)
        file_keys_class0 = list(files_by_class[0].keys())
        file_keys_class1 = list(files_by_class[1].keys())
        np.random.shuffle(file_keys_class0)
        np.random.shuffle(file_keys_class1)
        
        # Select samples for training
        train_indices = []
        test_indices = []
        
        # Track how many samples we've added to training for each class
        train_count = {0: 0, 1: 0}
        
        # First, let's add samples to the training set
        for cls in [0, 1]:
            file_keys = file_keys_class0 if cls == 0 else file_keys_class1
            
            for file_id in file_keys:
                # Shuffle indices within this file
                indices = files_by_class[cls][file_id]
                np.random.shuffle(indices)
                
                # How many more samples do we need for this class
                samples_needed = samples_per_class - train_count[cls]
                
                if samples_needed <= 0:
                    # We already have enough samples for this class
                    test_indices.extend(indices)
                elif len(indices) <= samples_needed:
                    # Take all samples from this file
                    train_indices.extend(indices)
                    train_count[cls] += len(indices)
                else:
                    # Take only what we need
                    train_indices.extend(indices[:samples_needed])
                    test_indices.extend(indices[samples_needed:])
                    train_count[cls] += samples_needed
        
        # Add any remaining samples to test
        for cls in [0, 1]:
            file_keys = file_keys_class0 if cls == 0 else file_keys_class1
            for file_id in file_keys:
                indices = files_by_class[cls][file_id]
                for idx in indices:
                    if idx not in train_indices and idx not in test_indices:
                        test_indices.append(idx)
        
        # Split the data
        X_train = X_all[train_indices]
        y_train = y_all[train_indices]
        X_test = X_all[test_indices]
        y_test = y_all[test_indices]
        
        print(f"Split summary:")
        print(f"  Train set: {X_train.shape[0]} samples ({np.sum(y_train == 0)} class 0, {np.sum(y_train == 1)} class 1)")
        print(f"  Test set: {X_test.shape[0]} samples ({np.sum(y_test == 0)} class 0, {np.sum(y_test == 1)} class 1)")
        # print(f"  Test files: {len(test_files)} out of {len(unique_files)} files")
        
        return X_train, y_train, X_test, y_test

    def train(self, root_dir: str):
        self.X_test_split = np.array([]) # Ensure these are reset/initialized
        self.y_test_split = np.array([])
        try:
            X_all, y_all, sample_metadata = self.load_dataset(root_dir)

            if X_all.shape[0] == 0:
                 print("Error: No data loaded from dataset. Model cannot be trained.")
                 self.model = None 
                 return
            
            # Use fixed split with 20 samples per class
            X_train, y_train, X_test, y_test = self.perform_fixed_split(X_all, y_all, sample_metadata, samples_per_class=20)

            self.X_test_split = X_test # Store for later evaluation
            self.y_test_split = y_test

            if X_train.shape[0] == 0:
                 print("Error: Training data is empty after split. Model cannot be trained.")
                 self.model = None
                 return
                
            self.model.fit(X_train, y_train)
            print(f"Logistic Regression model trained with {X_train.shape[0]} data points.")
            print(f"  Class 0: {np.sum(y_train == 0)} samples")
            print(f"  Class 1: {np.sum(y_train == 1)} samples")
            
            if self.X_test_split.shape[0] > 0:
                print(f"Test set has {self.X_test_split.shape[0]} data points.")
                print(f"  Class 0: {np.sum(y_test == 0)} samples")
                print(f"  Class 1: {np.sum(y_test == 1)} samples")
            else:
                print("No data points were held out for testing (possibly due to small class sizes).")

        except Exception as e:
            print(f"An error occurred during data loading, splitting, or training: {e}")
            self.model = None # Ensure model is marked as not trained on error

    def predict_from_file(self, file_path: str, expected_mat_var_name: str = None) -> list:
        all_predictions = []
        var_name_to_try = None
        try:
            mat_contents = scipy.io.loadmat(file_path)
            if expected_mat_var_name and expected_mat_var_name in mat_contents:
                var_name_to_try = expected_mat_var_name
            elif self.mat_var_name_class0 in mat_contents:
                var_name_to_try = self.mat_var_name_class0
            elif self.mat_var_name_class1 in mat_contents:
                var_name_to_try = self.mat_var_name_class1
            else:
                raise ValueError(f"Known data variables ('{self.mat_var_name_class0}', '{self.mat_var_name_class1}') not found in {os.path.basename(file_path)}.")
            print(f"  Predicting using variable '{var_name_to_try}' from {os.path.basename(file_path)}")
            data_cell_array = mat_contents[var_name_to_try]
            # ... (rest of prediction logic as before) ...
            if not (isinstance(data_cell_array, np.ndarray) and data_cell_array.dtype == 'object' and
                    data_cell_array.ndim == 2 and data_cell_array.shape[1] == 1):
                raise ValueError(f"Data cell array '{var_name_to_try}' in prediction file is malformed.")
            num_trials = data_cell_array.shape[0]
            if num_trials == 0: return []
            for i in range(num_trials):
                try:
                    trial_data_1d = data_cell_array[i, 0].squeeze()
                    if trial_data_1d.ndim != 1 or trial_data_1d.size == 0: continue
                    features = self.extract_features_from_single_channel(trial_data_1d)
                    if features.size > 0:
                        prediction = self.model.predict(features.reshape(1, -1))
                        all_predictions.append(int(prediction[0]))
                except Exception as trial_pred_e:
                    print(f"Error predicting trial {i+1} in {os.path.basename(file_path)}: {trial_pred_e}")
            return all_predictions
        except Exception as e:
            print(f"Error predicting from file {file_path}: {e}")
            return []

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model with more detailed metrics."""
        if not hasattr(self.model, "classes_") or getattr(self.model, 'classes_', None) is None:
            print("Model not trained. Cannot calculate metrics.")
            return {"error": "Model not trained"}
        if X_test.shape[0] == 0:
            print("Test data empty. Cannot calculate metrics.")
            return {"error": "Empty test data"}
        if X_test.shape[0] != y_test.shape[0]:
            print("X_test and y_test sample numbers mismatch.")
            return {"error": "Data shape mismatch"}
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "misclassification_error": 1 - accuracy_score(y_test, y_pred)
        }
        
        # Add class distribution
        metrics["class_distribution"] = {
            "test_set": {
                "class_0": int(np.sum(y_test == 0)),
                "class_1": int(np.sum(y_test == 1))
            },
            "predictions": {
                "class_0": int(np.sum(y_pred == 0)),
                "class_1": int(np.sum(y_pred == 1))
            }
        }
        
        return metrics

    def misclassification_error(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Legacy method maintained for compatibility"""
        if not hasattr(self.model, "classes_") or getattr(self.model, 'classes_', None) is None:
            print("Model not trained. Cannot calculate error.")
            return -1.0
        if X_test.shape[0] == 0:
            print("Test data empty. Cannot calculate error.")
            return -1.0 
        if X_test.shape[0] != y_test.shape[0]:
            print("X_test and y_test sample numbers mismatch.")
            return -1.0
        y_pred = self.model.predict(X_test)
        misclassified_count = np.count_nonzero(y_test - y_pred)
        error_rate = misclassified_count / len(y_test)
        return error_rate

if __name__ == '__main__':
    # === USER ACTION REQUIRED ===
    # These should be the names of the CELL ARRAY variables within your .mat files
    # that contain the ~5000x1 data segments for each class.
    MAT_VAR_CLASS_0 = 'trials'       # Variable name for cell array in relaxed .mat files (e.g., ./data2/class0/RelaxStimTrials.mat)
    MAT_VAR_CLASS_1 = 'StimContract' # Variable name for cell array in contracted .mat files (e.g., ./data2/class1/contractStim.mat)

    print(f"Config: Using MATLAB variable '{MAT_VAR_CLASS_0}' for class 0 data.")
    print(f"Config: Using MATLAB variable '{MAT_VAR_CLASS_1}' for class 1 data.")

    # Create multiple classifier instances with different random seeds to verify consistency
    random_seeds = [42, 123, 456]
    for seed in random_seeds:
        print(f"\n\n==== RUNNING WITH RANDOM SEED {seed} ====")
        classifier = LogisticRegressionClassifier(
            data_dir="./data2",     # Your data directory with class0/ and class1/ subfolders
            sr=10000,
            n_mfcc=13,
            logreg_max_iter=1000,
            mat_var_name_class0=MAT_VAR_CLASS_0,
            mat_var_name_class1=MAT_VAR_CLASS_1,
            random_state=seed
        )

        # Training happens in __init__ which calls self.train()
        # self.train() now populates self.X_test_split and self.y_test_split

        if hasattr(classifier.model, "classes_") and getattr(classifier.model, 'classes_', None) is not None:
            print("\n--- Evaluating Model on Held-Out Test Split ---")
            if classifier.X_test_split.shape[0] > 0 and classifier.y_test_split.shape[0] > 0:
                metrics = classifier.evaluate_model(classifier.X_test_split, classifier.y_test_split)
                
                if "error" not in metrics:
                    print(f"Test set size: {classifier.X_test_split.shape[0]} samples.")
                    print(f"Class distribution in test set: Class 0: {metrics['class_distribution']['test_set']['class_0']}, "
                          f"Class 1: {metrics['class_distribution']['test_set']['class_1']}")
                    print(f"Misclassification error: {metrics['misclassification_error']:.4f}")
                    print(f"Accuracy: {metrics['accuracy']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"F1 Score: {metrics['f1']:.4f}")
                    print(f"Confusion Matrix:")
                    print(f"  TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}")
                    print(f"  FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}")
                else:
                    print(f"Could not calculate metrics: {metrics['error']}")
            else:
                print("No test data was generated by the split.")
        else:
            print("\nModel was not trained successfully; skipping evaluation.")

    print("\n\nNOTE: If you're still getting very high accuracy (>95%) after these fixes, consider:")
    print("1. Your dataset might be too easily separable (the classes might be very distinct)")
    print("2. There might still be data leakage if multiple trials from the same source are too similar")
    print("3. Try examining specific misclassified examples to understand what makes them different")
    print("4. Consider cross-validation with file-based splits as a more robust evaluation")
    print("5. Remember that the original goal is 20 train samples per class, which means 6 class 0 samples and 4 class 1 samples in test based on your comment")