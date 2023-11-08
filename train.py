import matplotlib
import pandas as pd  # for data analysis
import numpy as np  # for calculation
import matplotlib.pyplot as plt  # for plotting graph
import seaborn as sns  # for plotting graphs
from jedi.api.refactoring import inline
from sklearn.feature_selection import SelectKBest  # for feature selection
from sklearn.feature_selection import f_classif, mutual_info_classif  # for feature selection
from sklearn.preprocessing import StandardScaler  # for feature scaling data to lower values
from sklearn.feature_extraction import DictVectorizer  # for one hot encoding
from imblearn.over_sampling import SMOTE  # for oversampling the data
from sklearn.svm import SVC  # 1st model
from sklearn.neighbors import KNeighborsClassifier  # 2nd model
from sklearn.naive_bayes import GaussianNB  # 3rd model
from sklearn.tree import DecisionTreeClassifier  # 4th model
from sklearn.ensemble import RandomForestClassifier  # 5th model
from sklearn.model_selection import train_test_split  # to split data to train and test
from sklearn.model_selection import \
    GridSearchCV  # one type of cross validation to find out best model and hyperparameters
from imblearn.pipeline import Pipeline  # model pipeline to prevent data leakage
from sklearn.metrics import classification_report  # to find precision, f1 score and recall
from sklearn.metrics import roc_auc_score  # AUC score
from sklearn.metrics import confusion_matrix  # to show the result in heatmap
import pickle  # to save the model

df = pd.read_csv('parkinsons.data')

print(df.head())

print(df.info())

print(df.describe())

df.drop('name', axis=1, inplace=True)

"""#Exploratory Data Analysis and Data Preprocessing"""

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x="status", hue='status', dodge=False)
plt.title('Distribution of Target Variable')
h, l = ax.get_legend_handles_labels()
labels = ["Healthy People", "People with Parkinson"]
ax.legend(h, labels, loc="upper left")
plt.show()

plt.figure(figsize=(10, 8))
columns_to_plot = df.columns.drop('status')
ax = sns.boxplot(data=df[columns_to_plot])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

print(df.isnull().sum())

df.drop_duplicates(inplace=True)

print(df.shape)

# outlier removal function
def remove_outliers_iqr(data, factor=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data

for column in ['MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)']:
    df[column] = remove_outliers_iqr(df[column])

print(df.shape)

print(df.isnull().sum())

print(df.describe())

# df = df[(df['MDVP:Fhi(Hz)'] <= 380)]
# df=df.dropna()

plt.figure(figsize=(10, 8))
columns_to_plot = df.columns.drop('status')
ax = sns.boxplot(data=df[columns_to_plot])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

# Set the number of columns per row in the plot grid
num_cols_per_row = 3

# Get the list of column names
columns = df.columns

# Calculate the number of rows needed to display all columns
num_rows = (len(columns) + num_cols_per_row - 1) // num_cols_per_row

# Create a figure and axes for the plots
fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(15, 15))
fig.tight_layout(pad=3.0)

# Iterate through the columns and plot the distributions
for i, column in enumerate(columns):
    ax = axes[i // num_cols_per_row, i % num_cols_per_row]
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(column)

# Remove any empty subplots if the number of columns is not a multiple of num_cols_per_row
for i in range(len(columns), num_rows * num_cols_per_row):
    fig.delaxes(axes[i // num_cols_per_row, i % num_cols_per_row])

# Show the plots
plt.show()

df = df.fillna(df.median())

print(df.isnull().sum())

columns_to_include = ['MDVP:Fo(Hz)', 'MDVP:Flo(Hz)', 'spread1', 'spread2', 'PPE', 'status']
sns.pairplot(df[columns_to_include], hue="status", markers=["o", "s"])
plt.show()

plt.figure(figsize=(8, 8))
numeric_columns = df.select_dtypes(include=['number'])
sns.heatmap(numeric_columns.corr().round(1), annot=True, vmax=1, vmin=-1, center=0, cmap='coolwarm', cbar=True,
            annot_kws={'size': 8})
plt.title('Correlation Matrix')
plt.show()

print(df.groupby('status').mean())

print(df['status'].value_counts())

X = df.drop('status', axis=1)
Y = df['status']

"""UNIVARIATE FEATURE SELECTION"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif

def featureSelect_dataframe(X, Y, criteria, k):

    reg = SelectKBest(criteria, k=k).fit(X,Y)
    X_transformed = reg.transform(X)
    selected_feature_indices = reg.get_support(indices=True)
    column_names = X.columns[selected_feature_indices].tolist()

    return print("The most important columns by feature selection method {} are: {}".format(criteria.__name__, f"{column_names}".strip('[]')))

feature_f = featureSelect_dataframe(X, Y, f_classif, 5)
feature_mutual = featureSelect_dataframe(X, Y, mutual_info_classif, 5)

"""# MODEL SELECTION, HYPERPARAMETERS TUNING, FEATURE SCALING & OVERSAMPLING

"""

X_train_ed, X_test_ed, Y_train_ed, Y_test_ed = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
X_train = X_train_ed.to_dict(orient='records')
X_test = X_test_ed.to_dict(orient='records')
Y_train = Y_train_ed.values
Y_test = Y_test_ed.values

print(X_train)

print(Y_train)

pipeline = Pipeline([
    ('dv', DictVectorizer(sparse=False)),  # for one hot encoding
    ('scaler', StandardScaler()),  # to scale the data
    # ('smote', SMOTE()),   # to oversample the minority data
    ('classifier', 'passthrough')  # Placeholder for the model
])

param_grid = [
    {
        'classifier': [SVC()],  # SVM model
        'classifier__C': [0.1, 1, 3, 5, 10],
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    },
    {
        'classifier': [KNeighborsClassifier()],  # kNN model
        'classifier__n_neighbors': [1, 3, 5, 7],
        'classifier__weights': ['uniform', 'distance']
    },
    {
        'classifier': [GaussianNB()],  # Gaussian Naive Bayes model
    },
    {
        'classifier': [DecisionTreeClassifier()],  # Decision Tree model
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__max_depth': [3, 6, 9],
        'classifier__max_leaf_nodes': [3, 6, 9],
    },
    {
        'classifier': [RandomForestClassifier()],  # Random Forest model
        'classifier__n_estimators': [25, 50, 100, 150],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__max_depth': [3, 6, 9],
        'classifier__max_leaf_nodes': [3, 6, 9],
    }
]

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, Y_train)

print('Best Model:', grid_search.best_estimator_)
print('highest score', grid_search.best_score_)
print('Best Hyperparameters:', grid_search.best_params_)

result_df = pd.DataFrame(grid_search.cv_results_)

pd.options.display.max_colwidth = 200
pd.set_option('display.max_rows', None)
columns = ['params', 'mean_test_score',
           'rank_test_score']  # to display the result of GridSearchCV: all the model corresponding it hyperparameters, test score and its ranking
print(result_df[columns])

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

pred_train = best_model.predict(X_train)

auc_train = roc_auc_score(Y_train, pred_train)

print(auc_train)

pred_test = best_model.predict(X_test)

prediction_df = pd.DataFrame({"Actual": Y_test, "Prediction": pred_test})

prediction_df.head()

auc_score = roc_auc_score(Y_test, pred_test)

print(auc_score)

print(classification_report(Y_train, pred_train))

print(classification_report(Y_test, pred_test))

cm = confusion_matrix(Y_test, pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, Y_train, cv=5,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

report = classification_report(Y_test, pred_test, output_dict=True)
class_names = list(report.keys())[:-3]
precision = []
recall = []
f1_score = []
for class_name in class_names:
    precision.append(report[class_name]['precision'])
    recall.append(report[class_name]['recall'])
    f1_score.append(report[class_name]['f1-score'])

# Plotting the performance metrics
x = np.arange(len(class_names))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

ax.set_ylabel('Score')
ax.set_title('Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend()

plt.show()

output_file = 'model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump(best_model, f_out)

print(f'The model is saved to {output_file}')