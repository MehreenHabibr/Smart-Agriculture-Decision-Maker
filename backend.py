#Imports/Packages
import io
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import label_binarize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import numpy as np
from scipy.stats import t, ttest_ind
from sklearn.metrics import (auc, confusion_matrix, log_loss, roc_curve)
import warnings
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
from flask import Flask, json, jsonify, request, send_file
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.utils import class_weight

# Reading the dataset
data = pd.read_csv('Crop and fertilizer dataset.csv')

# Using flask framework to build the web application
backend = Flask(__name__)
# Using CORS to allow request from different origins
CORS(backend)

data_sample = pd.read_csv('Crop and fertilizer dataset.csv')

# Function to send districts from dataset to appear in dropdown
@backend.route('/districts', methods=['POST','GET'])
def return_districts():
    districts = data_sample['District_Name'].tolist()  
    unique_districts = list(set(districts))
    return unique_districts

# Function to send soil colors from dataset to appear in dropdown
@backend.route('/soilcolor', methods=['POST','GET'])
def return_soilcolor():
    soil_color = data_sample['Soil_color'].tolist()  
    unique_soilcolor = list(set(soil_color))
    return unique_soilcolor

# Function to send fertilizers from dataset to appear in dropdown
@backend.route('/fertilizer', methods=['POST','GET'])
def return_fertilizer():
    fertilizer = data_sample['Fertilizer'].tolist()
    unique_fertilizer = list(set(fertilizer))
    return unique_fertilizer

# Plotting graph for target variables
@backend.route('/targetplot', methods=['POST','GET'])
def plot_targets():
    label_name = data['Crop'].value_counts().index
    val = data['Crop'].value_counts().values
    plt.figure(figsize=(6,8))
    plt.title('Distribution of Crops')
    plt.pie(x=val, labels=label_name, shadow=True, autopct='%1.1f%%')
    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Close the plot to free memory
    plt.close()
    # Return the image file to the frontend
    return send_file(buffer, mimetype='image/png')


# Encoding categorical features
label_encoders = {}
categorical_features = ['District_Name', 'Soil_color', 'Fertilizer']
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Splitting data into training and testing sets
X = data.drop(columns=['Crop', 'Link']).values
y = data['Crop'].values

# Training a Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Standardizing Features
sc = StandardScaler()

# Function to calculate error rates
def calculate_error_rate(predictions, true_labels):
    return np.sum(predictions != true_labels) / len(true_labels)
# Function to calculate the confidence interval
def calculate_confidence_interval(t_statistic, se_diff, df, confidence_level):
    alpha = 1 - confidence_level
    critical_value = t.ppf(1 - alpha / 2, df)
    margin_of_error = critical_value * se_diff
    lower_bound = t_statistic - margin_of_error
    upper_bound = t_statistic + margin_of_error
    return lower_bound, upper_bound
# Function to print the result for each pair of models
def print_result(t_statistic, p_value, model1, model2, error_diff, se_diff, df):
    confidence_level = 0.95
    conf_interval = calculate_confidence_interval(t_statistic, se_diff, df, confidence_level)
    if p_value < (1 - confidence_level) / 2:
        result = f"The error rate difference between {model1} and {model2} is statistically significant at {confidence_level * 100}% confidence level."
        if error_diff > 0:
            selected_model = model1
        elif error_diff < 0:
            selected_model = model2
        else:
            selected_model = model2
    else:
        result = f"The confidence interval contains 0; therefore, the difference in error rates between {model1} and {model2} may not be statistically significant at {confidence_level * 100}% confidence level."
    return conf_interval, result, selected_model
# Function that compares models
@backend.route('/models', methods=['GET'])
def model_comparisons():
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Precision loss occurred in moment calculation*")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Train models
    random_forest = RandomForestClassifier(random_state=42)
    svm = SVC(random_state=42)
    decision_tree = DecisionTreeClassifier(random_state=42)
    random_forest.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)
    # Make predictions on the test set
    random_forest_predictions = random_forest.predict(X_test)
    svm_predictions = svm.predict(X_test)
    decision_tree_predictions = decision_tree.predict(X_test)
    # Calculate error rates for each algorithm
    data['random_forest_error'] = calculate_error_rate(random_forest_predictions, y_test)
    data['svm_error'] = calculate_error_rate(svm_predictions, y_test)
    data['decision_tree_error'] = calculate_error_rate(decision_tree_predictions, y_test)
    # confidence_level = 0.95
    t_statistic_svm_rf, p_value_svm_rf = ttest_ind(data['svm_error'], data['random_forest_error'])
    t_statistic_svm_dt, p_value_svm_dt = ttest_ind(data['svm_error'], data['decision_tree_error'])
    t_statistic_rf_dt, p_value_rf_dt = ttest_ind(data['random_forest_error'], data['decision_tree_error'])
    # Perform t-tests for the difference in means
    svm_rf_ci, svm_rf_result, svm_rf_model = print_result(t_statistic_svm_rf, p_value_svm_rf, 'SVM', 'Random Forest',data['random_forest_error'].mean() - data['svm_error'].mean(),np.sqrt((data['random_forest_error'].var() / len(y_test)) + (data['svm_error'].var() / len(y_test))), len(y_test) - 1)
    svm_dt_ci, svm_dt_result, svm_dt_model = print_result(t_statistic_svm_dt, p_value_svm_dt, 'SVM', 'Decision Tree', data['decision_tree_error'].mean() - data['svm_error'].mean(), np.sqrt((data['decision_tree_error'].var() / len(y_test)) + (data['svm_error'].var() / len(y_test))), len(y_test) - 1)
    rf_dt_ci, rf_dt_result, rf_dt_model = print_result(t_statistic_rf_dt, p_value_rf_dt, 'Random Forest', 'Decision Tree', data['decision_tree_error'].mean() - data['random_forest_error'].mean(), np.sqrt((data['decision_tree_error'].var() / len(y_test)) + (data['random_forest_error'].var() / len(y_test))), len(y_test) - 1)
    # Create a dictionary to store the error rates and p-values for each model
    model_results = {
        'SVM': {'error_rate': data['svm_error'].mean(), 'p_value': p_value_svm_rf},
        'Random Forest': {'error_rate': data['random_forest_error'].mean(), 'p_value': p_value_rf_dt},
        'Decision Tree': {'error_rate': data['decision_tree_error'].mean(), 'p_value': p_value_svm_dt}
    }
    # Find the model with the lowest mean error rate
    overall_best_model = min(model_results, key=lambda model: model_results[model]['error_rate'])
    # Sending the results
    response_data = {
        'svm_rf_ci': svm_rf_ci,
        'svm_rf_result': svm_rf_result,
        'svm_rf_model': svm_rf_model,
        'svm_dt_ci': svm_dt_ci,
        'svm_dt_result': svm_dt_result,
        'svm_dt_model': svm_dt_model,
        'rf_dt_ci': rf_dt_ci,
        'rf_dt_result': rf_dt_result,
        'rf_dt_model': rf_dt_model,
        'best_model': overall_best_model
    }
    # Converting into a JSON response
    response = jsonify(response_data)
    # Adding CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')  
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')  
    return response

# Function to plot ROC curves
@backend.route('/roc', methods=['GET'])
def roc_curves():
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Initialize models after splitting the data
    random_forest = RandomForestClassifier(random_state=42)
    decision_tree = DecisionTreeClassifier(random_state=42)
    svm = SVC(random_state=42, probability=True)  # Set probability to True for SVM 
    # Train models
    random_forest.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)
    svm.fit(X_train, y_train) 
    # Assuming 'classes' is a list of unique class labels
    classes = np.unique(y_train)
    # Binarize the true labels
    y_true_bin = label_binarize(y_test, classes=classes)
    # Function to plot ROC curves for all models on a single graph
    def plot_roc_curves(models, model_names, y_true_bin, y_pred_probs):
        plt.figure(figsize=(5, 5))
        for i in range(len(models)):
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_probs[i].ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            # Plot ROC curve for each model
            plt.plot(fpr_micro, tpr_micro, lw=2, label=f'{model_names[i]} (AUC = {roc_auc_micro:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curves for Multiple Models')
        plt.legend(loc='lower right')
        plt.tight_layout()
        # Save the plot to a BytesIO object
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        plt.close()  # Close the plot to release memory
        return img_data
    # Plot ROC curves for all three models
    models = [random_forest, decision_tree, svm]
    model_names = ['Random Forest', 'Decision Tree', 'SVM']
    pred_probs = [random_forest.predict_proba(X_test), decision_tree.predict_proba(X_test), svm.predict_proba(X_test)]
    img_data = plot_roc_curves(models, model_names, y_true_bin, pred_probs)
    img_data.seek(0)
    return send_file(img_data, mimetype='image/png')

# Function to plot feature importance
@backend.route('/features', methods=['GET'])
def feature_importance():
    # Splitting the dataset and training the model
    random_forest = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    random_forest.fit(X_train, y_train)
    # Convert X_train to a DataFrame if it's not already
    X_train_df = pd.DataFrame(X_train, columns=['District_Name', 'Soil_color', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature', 'Fertilizer'])
    # Get feature importances
    importances = random_forest.feature_importances_
    features = X_train_df.columns
    # Plot feature importances
    plt.figure(figsize=(12,6))
    plt.barh(features, importances, align='center')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance using Random Forest')
    plt.gca().invert_yaxis()
    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Close the plot to free memory
    plt.close()
    # Return the image file to the frontend
    return send_file(buffer, mimetype='image/png')

# Implementing random forest from scratch
class RandomForest(BaseEstimator):
    def __init__(self, n_trees=7, max_depth=13, min_samples=2, min_samples_leaf=1, min_samples_split = 2, n_estimators =100, criterion =log_loss, bootstrap=True):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.min_samples = min_samples
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.criterion= criterion
        self.bootstrap = bootstrap
    def fit(self, X, y):
        self.trees = []
        # Encode the target variable if it's not numeric
        if not np.issubdtype(y.dtype, np.number):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
        # Calculate class weights
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
        class_weights = dict(enumerate(weights))
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                          min_samples_split=self.min_samples,
                                          class_weight=class_weights)
            dataset_sample = self.bootstrap_samples(X, y_encoded)
            X_sample, y_sample = dataset_sample[:, :-1], dataset_sample[:, -1]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self
    def predict_proba(self, X):
        # Get predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])     
        # Calculate class probabilities
        proba = np.mean(tree_preds, axis=0)
        return proba
    def bootstrap_samples(self, X, y):
        unique_classes = np.unique(y)
        samples_per_class = len(y) // len(unique_classes)
        sampled_indices = np.hstack([
            np.random.choice(np.where(y == uc)[0], samples_per_class, replace=True) 
            for uc in unique_classes
        ])
        np.random.shuffle(sampled_indices)
        return np.concatenate((X[sampled_indices], y[sampled_indices].reshape(-1, 1)), axis=1)
    def most_common_label(self, y):
        y = list(y)
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        preds = np.swapaxes(predictions, 0, 1)
        majority_predictions = np.array([self.most_common_label(pred) for pred in preds])
        # Decode the predictions if label encoder is used
        if self.label_encoder:
            majority_predictions = self.label_encoder.inverse_transform(majority_predictions.astype(int))
        return majority_predictions
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def get_params(self, deep=True):
        return {"n_trees": self.n_trees, "max_depth": self.max_depth, "min_samples": self.min_samples}
    def score(self, X, y):
        # Predict using the RandomForest model
        predictions = self.predict(X)
        # Calculate accuracy
        accuracy = np.mean(predictions == y)
        return accuracy
    
# scaling
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Function to predict the crop
def recommendation(District_Name, Soil_color, Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, Fertilizer, rmodel, sc, label_encoders):
    # Encode the categorical features
    District_Name_encoded = label_encoders['District_Name'].transform([District_Name])[0]
    Soil_color_encoded = label_encoders['Soil_color'].transform([Soil_color])[0]
    Fertilizer_encoded = label_encoders['Fertilizer'].transform([Fertilizer])[0]
    # Arrange features into a numpy array
    features = np.array([[District_Name_encoded, Soil_color_encoded, Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, Fertilizer_encoded]])
    # X = sc.fit(features)
    # Scale features using StandardScaler
    X = sc.transform(features)
    # Predict the crop
    prediction = rmodel.predict(X)[0]
    return prediction

# Initialize the RandomForestClassifier
random_forest = RandomForest(
    max_depth=10, 
    min_samples_leaf=3, 
    min_samples_split=7, 
    n_estimators=100, 
    criterion='gini',
    bootstrap=True,
    n_trees=77
)
# Cross-validation predictions
y_pred = cross_val_predict(random_forest, X_train, y_train, cv=10)
# Compute the Confusion Matrix
cm = confusion_matrix(y_train, y_pred)

# Function to print predicted crop and evaluate metrics
@backend.route('/', methods=['POST'])
def recommend():
    # Handling incoming HTTP requests
    content_type = request.headers.get('Content-Type')
    if content_type != 'application/json':
        return "Unsupported Media Type", 415  # Unsupported Media Type status code
    result = None
    if request.method == 'POST':
        # Get the JSON data from the request
        json_data = request.get_json()
        # Access individual fields from the JSON data
        district_name = json_data.get('distname')
        soil_color = json_data.get('soilcolor')
        nitrogen = json_data.get('nitrogen')
        phosphorus = json_data.get('phosphorus')
        potassium = json_data.get('potassium')
        ph = json_data.get('pH')
        rainfall = json_data.get('rainfall')
        temp = json_data.get('temp')
        fertilizer = json_data.get('fertilizer')
        #Training the model
        rmodel = RandomForest(9,10,4)
        rmodel.fit(X_train, y_train)
        # Call the recommendation function with the retrieved values
        result = recommendation(
            district_name, soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temp, fertilizer, rmodel, sc, label_encoders
        )
        # Extract the metrics for the specific crop
        crop_index = list(np.unique(y_train)).index(result)
        true_positives = cm[crop_index][crop_index]
        false_positives = sum(cm[:, crop_index]) - true_positives
        false_negatives = sum(cm[crop_index, :]) - true_positives
        true_negatives = sum(sum(cm)) - (true_positives + false_positives + false_negatives)
        # Compute metrics
        accuracy_crop = (true_positives + true_negatives) / sum(sum(cm))
        precision_crop = true_positives / (true_positives + false_positives)
        recall_crop = true_positives / (true_positives + false_negatives)
        f1_crop = 2 * (precision_crop * recall_crop) / (precision_crop + recall_crop)
        # Sending the results
        response_data = {
            'result': result,
            'accuracy': accuracy_crop,
            'precision': precision_crop,
            'recall': recall_crop,
            'f1': f1_crop
        }
        # Converting into a JSON response
        response = json.dumps(response_data)
        return response
    return "Invalid request method", 400 

















