
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split # Not used in the class directly, but good for testing
from sklearn.metrics import classification_report # Not used in the class, but good for testing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt # Not used in the class, but good for visualization

class KNNClassifier:
    def __init__(self, data_dir="./data", sr=10000, n_mfcc=13, n_neighbors=1):
        """
        Initializes the KNNClassifier.

        Args:
            data_dir (str): The root directory where the training data is stored.
                            It should contain subdirectories for each class (e.g., "class0", "class1").
            sr (int): The target sampling rate for audio loading.
            n_mfcc (int): The number of MFCCs to extract.
            n_neighbors (int): The number of neighbors for the KNN algorithm.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.sr = sr
        self.n_mfcc = n_mfcc

        # Train the model upon initialization
        if data_dir:
            print(f"Training model with data from: {data_dir}")
            self.train(data_dir)
            print("Model training complete.")
        else:
            print("Warning: No data directory provided for training. Model is not trained.")


    def extract_features(self, file_path: str) -> np.ndarray:
        """
        Extracts features from a 2-channel audio file.
        Features consist of the mean and standard deviation of MFCCs from each channel,
        concatenated together.

        Args:
            file_path (str): Path to the .wav audio file.

        Returns:
            np.ndarray: A 1D numpy array of extracted features.
                        The length will be 4 * n_mfcc.

        Raises:
            ValueError: If the audio file is not 2-channel.
            Exception: For other librosa or numpy errors during feature extraction.
        """
        try:
            # Load stereo audio. mono=False ensures we get both channels if available.
            y_stereo, sample_rate = librosa.load(file_path, sr=self.sr, mono=False)

            # Validate that the audio is indeed 2-channel
            if y_stereo.ndim != 2 or y_stereo.shape[0] != 2:
                raise ValueError(f"Audio file {os.path.basename(file_path)} is not 2-channel. Detected shape: {y_stereo.shape}")

            y_channel1 = y_stereo[0, :]  # EMG data
            y_channel2 = y_stereo[1, :]  # Stimulation data

            # pruned_emg_channel = np.array([])
            # pruned_stim_channel = np.array([])

            # indices_of_nonzero = np.nonzero(y_channel1 != 0)[0]

            # if indices_of_nonzero.size > 0:
            #     first_nonzero_index = indices_of_nonzero[0]

            #     pruned_emg_channel_step1 = y_channel1[first_nonzero_index:]
            #     pruned_stim_channel_step1 = y_channel2[first_nonzero_index:]

            #     indices_of_zero_in_pruned_emg_step1 = np.where(pruned_emg_channel_step1 == 0)[0]

            #     if indices_of_zero_in_pruned_emg_step1.size > 0:
            #         first_zero_index_in_pruned = indices_of_zero_in_pruned_emg_step1[0]
            #         pruned_emg_channel = pruned_emg_channel_step1[:first_zero_index_in_pruned]
            #         pruned_stim_channel = pruned_stim_channel_step1[:first_zero_index_in_pruned]
            #     else:
            #         pruned_emg_channel = pruned_emg_channel_step1
            #         pruned_stim_channel = pruned_stim_channel_step1


            extracted_emg_segments = []
            extracted_stim_segments = []

            current_search_offset = 0

            for _ in range(21):
                if current_search_offset >= len(y_channel1):
                    break

                active_y_channel1 = y_channel1[current_search_offset:]
                active_y_channel2 = y_channel2[current_search_offset:]

                if active_y_channel1.size == 0: # No more data to process
                    break

                indices_of_activity_start_in_view = np.nonzero(active_y_channel1 != 0)[0]

                if indices_of_activity_start_in_view.size > 0:
                    segment_start_in_view = indices_of_activity_start_in_view[0]

                    # This is the segment of interest from the first non-zero point in the current view
                    current_emg_activity_view = active_y_channel1[segment_start_in_view:]
                    current_stim_activity_view = active_y_channel2[segment_start_in_view:]
                    
                    # Find the first zero *within this new active segment view*
                    indices_of_silence_in_active_segment = np.where(current_emg_activity_view == 0)[0]

                    segment_emg_to_add = np.array([])
                    segment_stim_to_add = np.array([])
                    
                    advance_by_in_original = 0

                    if indices_of_silence_in_active_segment.size > 0:
                        # A zero was found, marking the end of the current non-zero segment
                        segment_end_in_activity_view = indices_of_silence_in_active_segment[0]
                        
                        segment_emg_to_add = current_emg_activity_view[:segment_end_in_activity_view]
                        segment_stim_to_add = current_stim_activity_view[:segment_end_in_activity_view]
                        
                        # Advance offset to start after the found segment and the zero that ended it
                        advance_by_in_original = segment_start_in_view + segment_end_in_activity_view + 1
                    else:
                        # No zero found after non-zero activity, so this segment runs to the end of the current view
                        segment_emg_to_add = current_emg_activity_view
                        segment_stim_to_add = current_stim_activity_view
                        
                        # Advance offset to start after this entire segment
                        advance_by_in_original = segment_start_in_view + len(current_emg_activity_view)
                    
                    # Only add non-empty segments
                    if segment_emg_to_add.size > 0:
                        extracted_emg_segments.append(segment_emg_to_add)
                        extracted_stim_segments.append(segment_stim_to_add)

                    current_search_offset += advance_by_in_original

                else:
                    break

            pruned_emg_channel = np.concatenate(extracted_emg_segments)
            pruned_stim_channel = np.concatenate(extracted_stim_segments)

            # Extract MFCCs for Channel 1 (EMG)
            mfcc_ch1 = librosa.feature.mfcc(y=pruned_emg_channel, sr=sample_rate, n_mfcc=self.n_mfcc)
            # mfcc_ch1 = librosa.feature.mfcc(y=y_channel1, sr=sample_rate, n_mfcc=self.n_mfcc)
            mean_mfcc_ch1 = np.mean(mfcc_ch1, axis=1)
            std_mfcc_ch1 = np.std(mfcc_ch1, axis=1)

            # Extract MFCCs for Channel 2 (Stimulation)
            mfcc_ch2 = librosa.feature.mfcc(y=pruned_stim_channel, sr=sample_rate, n_mfcc=self.n_mfcc)
            # mfcc_ch2 = librosa.feature.mfcc(y=y_channel2, sr=sample_rate, n_mfcc=self.n_mfcc)
            mean_mfcc_ch2 = np.mean(mfcc_ch2, axis=1)
            std_mfcc_ch2 = np.std(mfcc_ch2, axis=1)

            # Concatenate features: [mean_ch1, std_ch1, mean_ch2, std_ch2]
            # This combined vector allows the KNN to find patterns based on both channels' characteristics
            # and their interplay as represented in this feature space.
            features = np.concatenate([mean_mfcc_ch1, std_mfcc_ch1, mean_mfcc_ch2, std_mfcc_ch2])
            
            return features

        except Exception as e:
            # Re-raise the exception to be caught by load_dataset or other calling functions
            raise Exception(f"Error extracting features from {os.path.basename(file_path)}: {e}")


    def load_dataset(self, root_dir: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the dataset from the given root directory.
        Assumes root_dir contains subdirectories named "class0", "class1", etc.,
        each containing .wav files for that class.

        Args:
            root_dir (str): The root directory of the dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X: A numpy array of feature vectors (samples x features).
                - y: A numpy array of corresponding labels.
        """
        X, y_labels = [], []
        
        # Simple binary classification for now: class0 -> 0, class1 -> 1
        # This can be extended for multi-class by creating a mapping.
        class_to_label = {}
        label_counter = 0

        print(f"Loading dataset from {root_dir}...")
        for label_dir_name in sorted(os.listdir(root_dir)): # Sort for consistent label assignment
            full_dir_path = os.path.join(root_dir, label_dir_name)
            if not os.path.isdir(full_dir_path):
                continue

            if label_dir_name not in class_to_label:
                # Assign a new integer label for each new class directory found
                # Forcing specific names for binary case as per original code
                if label_dir_name == "class0":
                    class_to_label[label_dir_name] = 0
                elif label_dir_name == "class1":
                     class_to_label[label_dir_name] = 1
                else:
                    # For other classes, assign dynamically if needed, or restrict
                    print(f"Warning: Directory '{label_dir_name}' does not match 'class0' or 'class1'. It will be skipped unless logic is updated.")
                    continue # Skip if not class0 or class1 for strict binary
            
            current_label = class_to_label[label_dir_name]
            print(f"Processing directory: {label_dir_name}, assigned label: {current_label}")

            for fname in os.listdir(full_dir_path):
                if fname.lower().endswith(".wav"):
                    fpath = os.path.join(full_dir_path, fname)
                    try:
                        features = self.extract_features(fpath)
                        X.append(features)
                        y_labels.append(current_label)
                    except ValueError as ve: # Catch specific ValueError from extract_features
                        print(f"Skipping file {fpath} due to ValueError: {ve}")
                    except Exception as e:
                        print(f"Error processing file {fpath}, skipping: {e}")
        
        if not X:
            raise ValueError("No features were extracted. Check data directory and file formats.")
            
        return np.array(X), np.array(y_labels)

    def train(self, root_dir: str):
        """
        Trains the KNN model using data from the specified root directory.

        Args:
            root_dir (str): The root directory of the training dataset.
        """
        X_train, y_train = self.load_dataset(root_dir)
        if X_train.size == 0 or y_train.size == 0:
            print("Error: Training data is empty. Model cannot be trained.")
            return
        self.model.fit(X_train, y_train)
        print(f"Model trained with {X_train.shape[0]} samples from {root_dir}.")

    def predict(self, file_path: str) -> int:
        """
        Predicts the class label for a single audio file.

        Args:
            file_path (str): Path to the .wav audio file.

        Returns:
            int: The predicted class label.
        
        Raises:
            Exception: If feature extraction fails for the given file.
        """
        try:
            features = self.extract_features(file_path)
            # Reshape because scikit-learn expects a 2D array for a single sample
            prediction = self.model.predict(features.reshape(1, -1))
            return int(prediction[0])
        except Exception as e:
            print(f"Error predicting for file {file_path}: {e}")
            raise # Re-raise after logging or handle as appropriate


    def misclassification_error(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculates misclassification error (the rate at which this model is making incorrect predictions of y).
        Note that `misclassification_error = 1 - accuracy`.

        Args:
            X_test (np.ndarray): Test observations represented as `(n, d)` matrix.
                                 n is number of observations, d is number of features.
            y_test (np.ndarray): True target labels represented as `(n, )` vector.
                                 n is number of observations.

        Returns:
            float: Percentage of times prediction did not match target (misclassification error).
                   Returns -1.0 if the model is not trained or inputs are invalid.
        """
        if not hasattr(self.model, "classes_"):
            print("Model is not trained yet. Cannot calculate misclassification error.")
            return -1.0
        if X_test.shape[0] == 0:
            print("Test data is empty. Cannot calculate misclassification error.")
            return -1.0
        if X_test.shape[0] != y_test.shape[0]:
            print("Number of samples in X_test and y_test do not match.")
            return -1.0

        # Use the model's predict method for batch predictions
        y_pred = self.model.predict(X_test)
        
        # Count non-zero elements in the difference array (where y_true != y_pred)
        misclassified_count = np.count_nonzero(y_test - y_pred)
        
        # Calculate misclassification error
        error_rate = misclassified_count / len(y_test)
        
        return error_rate

if __name__ == '__main__':
    classifier = KNNClassifier(data_dir="./data", sr=10000, n_mfcc=13, n_neighbors=1)

    if classifier.model and hasattr(classifier.model, "classes_"): # Check if model is trained
        try:
            X_eval, y_eval = classifier.load_dataset("./data") # Using training data for eval demo
            if X_eval.shape[0] > 0:
                error = classifier.misclassification_error(X_eval, y_eval)
                accuracy = 1 - error
                print(f"Misclassification error on './data': {error:.4f}")
                print(f"Accuracy on './data': {accuracy:.4f}")

                # You could also use sklearn.metrics.classification_report
                # y_pred_eval = classifier.model.predict(X_eval)
                # print("\nClassification Report (on dummy './data'):")
                # print(classification_report(y_eval, y_pred_eval, target_names=["class0", "class1"]))

            else:
                print("No data loaded for evaluation example.")
        except ValueError as ve:
                print(f"Could not load data for evaluation: {ve}")
        except Exception as e:
            print(f"An error occurred during evaluation example: {e}")
    else:
        print("Model not trained, skipping misclassification error calculation.")

