# coding: utf-8
#%% importing the main modules used

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble

# The lines below display the versions of the libraries

print(np.__version__)
print(pd.__version__)
print(matplotlib.__version__)
print(sklearn.__version__)

#%%
# importing data

PATH       = 'C:/Users/geffa/PycharmProjects/pythonProject/Codigo_Python_Fabio/'

Filenames  = 'dataset.csv'

df_dataset = pd.read_csv(f'{PATH}{Filenames}',
                         encoding = 'UTF-8', sep = ',', low_memory = False)

#%%
# Authors: Fábio A. N. Setúbal <fabioans@ufpa.br>

# Separating the data into X (predictor variables) and y (predicted variables).

X = df_dataset.drop(['Point', 'Amplitude','Frequency', 'X', 'Y'], axis=1)

y = df_dataset[['Point', 'Amplitude','Frequency']]

# X, y = df_raw.drop(['Point', 'Amplitude','Frequency', 'X', 'Y'], axis=1), df_raw['Point']
# X, y = df_raw.drop(['Point', 'Amplitude','Frequency', 'X', 'Y'], axis=1), df_raw['Amplitude']
# X, y = df_raw.drop(['Point', 'Amplitude','Frequency', 'X', 'Y'], axis=1), df_raw['Frequency']
# X, y = df_raw.drop(['Point', 'Amplitude','Frequency', 'X', 'Y'], axis=1), df_raw[['Amplitude','Frequency', 'X', 'Y']]

# Above are being considered: the point of force application, the amplitude
# and the frequency. Your X and Y coordinates were NOT considered.

#%%
# Randomly separating 1% of the dataset data, for later validation.

# To perform separation of the df_dataset (X, y) set into training data
# and test data, the train_test_split function will be used:

# Before that, 1% of the X and y data will be reserved for model validation.
# This data will not be used during the optimization step.

X_train, X_validation, y_train, y_validation = sklearn.model_selection.train_test_split(X, y, test_size = 0.01, random_state = 0)

# In the above line, a dataset is separated for final validation.
# This dataset will not be used during the training and testing steps.
# This dataset is randomly extracted from the initial dataset.

#%%
# Separating data for training and testing.

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

# The sklearn.model_selection.train_test_split() function separates data into training and testing in a random way.
# In the above case, a fraction of 20% of the data is reserved for testing.

# Note: the random_state argument specifies the seed of generation of
# pseudorandomness of the algorithm; If default, data is randomized on each run
# If an integer is entered, the data is randomized but repeats across runs
# future. For testing this is fine, but in practice, you should leave the default.

#%%
# The function below accepts a DataFrame and plots a dendrogram type graph
# showing Spearman correlations between the variables:

def dendogram_spearmanr(df, tags):

    import scipy.cluster.hierarchy
    import scipy.stats
    
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = scipy.cluster.hierarchy.distance.squareform(1-corr)
    z = scipy.cluster.hierarchy.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(30,15))
    dendrogram = scipy.cluster.hierarchy.dendrogram(z, labels=tags, orientation='left', leaf_font_size=16)
    plt.show()

#%%    
dendogram_spearmanr(X_train, X_train.columns)

#%%
# To save time, a function will be defined, called display_score,
# which accepts a trained model and prints the metrics to the screen
#  MAE, MSE, RMSE and R2 related to training and testing:

def mae(y_true, y_pred):
    
    return sklearn.metrics.mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred): 
    
    return sklearn.metrics.mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred): 
    
    return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))

def r2(y_true, y_pred): 
    
    return sklearn.metrics.r2_score(y_true, y_pred)


def display_score(m):
    
    results = [[mae( y_train, m.predict(X_train)),
                mse( y_train, m.predict(X_train)),
                rmse(y_train, m.predict(X_train)),
                r2(  y_train, m.predict(X_train))],
               [mae( y_test, m.predict(X_test)),
                mse( y_test, m.predict(X_test)),
                rmse(y_test, m.predict(X_test)),
                r2(  y_test, m.predict(X_test))]]
    
    score = pd.DataFrame(results, columns=['MAE','MSE','RMSE','R2'],
                         index = ['Treino','Teste'])
    
    if hasattr(m, 'oob_score_'): 
        score.loc['OOB'] = [mae(y_train, m.oob_prediction_),
                            mse(y_train, m.oob_prediction_),
                            rmse(y_train, m.oob_prediction_),
                            m.oob_score_]
    display(score)

# Out-of-bag score

# Due to replacement sampling, each tree ignores an observation plot.

# The OOB (out-of-bag) metric uses this fact to measure the capacity
# predictive model without the need for a separate test suite.

# To make predictions and calculate OOB, each tree uses
# training data that it ignored.
# As the tree did not train the model with this data,
# they effectively work as a good test suite!

# In scikit-learn, you need to provide the parameter oob_score = True
# so that the OOB is calculated during training.

#%%
# Function to view a tree:

# draw_tree(m_tree.estimators_[0], X_train, precision=3)

def draw_tree(t, df, size=10, ratio=1, precision=0):
   
    import re
    import graphviz
    import sklearn.tree
    import IPython.display
    
    s=sklearn.tree.export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                                   special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))

#%% ===========================================================================
# ============= FIRST TEST: RFR DEFAULT ============================++++=======

# The regression model chosen will be the random model of forests.
# It will be imported from scikit-learn, stored in an object called "model_RFR".

# RFR MODEL PARAMERS

n_estimators             = 100   # (n_estimators             = 100   ) default
criterion                = 'mse' # (criterion                = 'mse' ) default 
max_depth                = None  # (max_depth                = None  ) default
min_samples_split        = 2     # (min_samples_split        = 2     ) default
min_samples_leaf         = 1     # (min_samples_leaf         = 1     ) default
min_weight_fraction_leaf = 0.0   # (min_weight_fraction_leaf = 0.0   ) default
max_features             = 0.5   # (max_features             = 'auto') default
max_leaf_nodes           = None  # (max_leaf_nodes           = None  ) default
min_impurity_decrease    = 0.0   # (min_impurity_decrease    = 0.0   ) default
min_impurity_split       = None  # (min_impurity_split       = None  ) default
bootstrap                = True  # (bootstrap                = True  ) default
oob_score                = True  # (oob_score                = False ) default
n_jobs                   = -1    # (n_jobs                   = None  ) default
random_state             = 0     # (random_state             = None  ) default
verbose                  = 0     # (verbose                  = 0     ) default
warm_start               = False # (warm_start               = False ) default
ccp_alpha                = 0.0   # (ccp_alpha                = 0.0   ) default
max_samples              = None  # (max_samples              = None  ) default

model_RFR = sklearn.ensemble.RandomForestRegressor(n_estimators             = n_estimators,
                                                    criterion                = criterion,
                                                    max_depth                = max_depth,
                                                    min_samples_split        = min_samples_split,
                                                    min_samples_leaf         = min_samples_leaf,
                                                    min_weight_fraction_leaf = min_weight_fraction_leaf,
                                                    max_features             = max_features,
                                                    max_leaf_nodes           = max_leaf_nodes,
                                                    min_impurity_decrease    = min_impurity_decrease,
                                                    min_impurity_split       = min_impurity_split,
                                                    bootstrap                = bootstrap,
                                                    oob_score                = oob_score,
                                                    n_jobs                   = n_jobs,
                                                    random_state             = random_state,
                                                    verbose                  = verbose,
                                                    warm_start               = warm_start,
                                                    ccp_alpha                = ccp_alpha,
                                                    max_samples              = max_samples)
                                                    
#%%
%time model_RFR.fit(X_train, y_train)

y_test_pred = model_RFR.predict(X_test)

display_score(model_RFR)

#%%
plt.plot(y_test, y_test_pred,'.')

# plotting the line x=y
plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim())

# axis legend
plt.xlabel('y_test')
plt.ylabel('y_test_pred')

# The line in the figure above is not the model! It's just the line x=y.
# The model cannot be visualized in the Cartesian plane, since it is multidimensional.
# The random forest model is non-linear, so it would not take the form of a straight line.


# Importance of variables

# The random forest model internally calculates a ranking
# of importance of the variables.
# For a given variable, the greater the decrease in error in splits of
# decisions made based on that variable, the more important it will be.

# This ranking is stored in the model's feature_importances_ attribute.

# Below, a function will be created that accepts a model and a list
# with the names of the variables,
# prints information regarding the importance ranking on the screen
# and returns a DataFrame with the ranking itself.
#%%
def plot_importances(model, tags, n=10):
    
    fig, ax = plt.subplots(1,2, figsize = (20,4))

    coefs = []
    abs_coefs = []

    if hasattr(model,'coef_'):
        imp = model.coef_
    elif hasattr(model,'feature_importances_'):
        imp = model.feature_importances_
    else:
        print('Set the model's coef or feature_importances!')
        return

    coefs = (pd.Series(imp, index = tags))
    coefs.plot(use_index=False, ax=ax[0]);
    abs_coefs = (abs(coefs)/(abs(coefs).sum()))
    abs_coefs.sort_values(ascending=False).plot(use_index=False, ax=ax[1],marker='.')

    ax[0].set_title('Relative importance of variables')
    ax[1].set_title('Relative importance of variables - descending order')

    abs_coefs_df = pd.DataFrame(np.array(abs_coefs).T,
                                columns = ['Importances'],
                                index = tags)

    df = abs_coefs_df['Importances'].sort_values(ascending=False)
    
    print(df.iloc[0:n])
    plt.figure()
    df.iloc[0:n].plot(kind='barh', figsize=(15,0.25*n), legend=False)
    
    return df

#%%
# Using the function to analyze the importance of the model:

imp = plot_importances(model_RFR, X_test.columns,49)

#%%
# Removing unimportant variables

# Unimportant variables can be discarded,
# which perhaps improves the model's accuracy and, certainly, the computational performance.

# # Selecting, for example, only those with more than 1% importance:
to_keep = imp[imp>0.019589852518983056].index
print(to_keep.shape)
#%%
X_train_keep = X_train[to_keep]
X_test_keep = X_test[to_keep]

#%% How many irrelevant variables were eliminated?
def display_score_keep(m):
    
    results = [[mae( y_train, m.predict(X_train_keep)),
                mse( y_train, m.predict(X_train_keep)),
                rmse(y_train, m.predict(X_train_keep)),
                r2(  y_train, m.predict(X_train_keep))],
               [mae( y_test, m.predict(X_test_keep)),
                mse( y_test, m.predict(X_test_keep)),
                rmse(y_test, m.predict(X_test_keep)),
                r2(  y_test, m.predict(X_test_keep))]]
    
    score = pd.DataFrame(results, columns=['MAE','MSE','RMSE','R2'],
                         index = ['Treino','Teste'])
    
    if hasattr(m, 'oob_score_'): 
        score.loc['OOB'] = [mae(y_train, m.oob_prediction_),
                            mse(y_train, m.oob_prediction_),
                            rmse(y_train, m.oob_prediction_),
                            m.oob_score_]
    display(score)
#%%
# Correlation analysis

# A correlation analysis is useful to understand the relationships between variables.
# The most used correlation for this is the Pearson correlation,
# that measures the degree of linear association between variables.
# Two variables are linearly associated if changes in one variable
# imply proportional changes in the other variable.
# https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de_Pearson

# In this code, the Spearman correlation will be used,
# which measures the degree of monotonic association between the variables.
# Two variables are monotonically associated if changes in one variable
# imply changes in the same direction (increasing or decreasing) in the other variable.
# It is a more general conception of association than Pearson's.
# https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de_postos_de_Spearman

#%%
# The graph is generated using hierarchical clustering techniques,
# that separated the variables into groups according to the correlations between them.
# If it is evident that some variables have very high correlation, this
# means they have the same information and are potentially redundant.
# https://en.wikipedia.org/wiki/Hierarchical_clustering

# The next cells can be used to remove some variables
# that the graph can indicate as redundant and after removal, it can be
# check the effect on OOB.
# If the effect is small, you can discard the variables.

#%%
# Removing redundant variables

# The function below is defined to speed up the analysis:
   # it accepts a set X, performs a training and returns the OOB score.

def get_oob(X):
    m = sklearn.ensemble.RandomForestRegressor(n_estimators = 100, min_samples_leaf = 1, 
                                               max_features = 0.5, n_jobs=-1, max_samples = None,
                                               oob_score = True, random_state = 0)
    m.fit(X, y_train)
    return m.oob_score_

#%%
# Setting the reference:
print(get_oob(X_train_keep))

#%% Train, predict and evaluate the model with the remaining features

%time model_RFR.fit(X_train_keep, y_train)
y_test_pred_keep = model_RFR.predict(X_test_keep)
display_score_keep(model_RFR)

#%%
plt.plot(y_test, y_test_pred_keep,'.')

# plotting the line x=y
plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim())

# axes legend
plt.xlabel('y_test')
plt.ylabel('y_test_pred_keep')

# The line in the figure above is not the model! It's just the line x=y.
# The model cannot be visualized in the Cartesian plane, since it is multidimensional.
# The random forest model is non-linear, so it would not take the form of a straight line.
#%%
# Performing analysis with removal of a potentially redundant variable
# at a time:
for c in ('F40', 'F41', 'F10', 'F47', 'F9',
          'F12', 'F15', 'F2', 'F5', 'F17'):
    print(c, get_oob(X_train_keep.drop(c, axis=1)))

#%%
# If the OOB does not decrease significantly, then none of these variables
# will seem to be missed!

# Performing the actual removals:

to_drop = ['F40', 'F41', 'F10', 'F47', 'F9',
          'F12', 'F15', 'F2', 'F5', 'F17']
print(get_oob(X_train_keep.drop(to_drop, axis=1)))
X_train_keep = X_train_keep.drop(to_drop, axis=1) 

print(X_train_keep.shape)


# With this procedure, it is possible to further reduce the number of variables.
# ============================================================================
#=================================== THE END =================================
# ============================================================================
#%%
   
    
    






































































































    






   







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




















    
    
    
    
    













