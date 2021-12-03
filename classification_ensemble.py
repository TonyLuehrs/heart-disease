import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFECV

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

###############################################################################################
#
#  This code is a lightly modified version of the code from Frank Ceballos at
#
#  https://github.com/frank-ceballos/Model_Design-Selection
#
#  or
#
#  https://towardsdatascience.com/model-design-and-selection-with-scikit-learn-18a29041d02a
#
###############################################################################################

class PipelineRFE(Pipeline):
    # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self


def get_feature_importances(X_train, y_train):
    
    ######################
    # Intended Usage:
    #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)
    #    feat_imps, sel_cols = get_feature_importances(X_train, y_train)
    ######################
    
    ######################
    # Step 1 - Use GridSearchCV() to find optimal hyperparameters for a Random Forest
    ######################
    
    rf = RandomForestClassifier()
    scaler = StandardScaler()
    pipeline = Pipeline(steps=[("scaler", scaler), ("classifier", rf)])

    rf_param_grid = {"classifier__n_estimators": [200],
                     "classifier__class_weight": [None, "balanced"],
                     "classifier__max_features": ["auto", "sqrt", "log2"],
                     "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                     "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                     "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                     "classifier__criterion" :["gini", "entropy"],
                     "classifier__n_jobs": [-1]}

    gscv = GridSearchCV(pipeline, rf_param_grid, cv=5, n_jobs=-1, verbose=1, scoring="roc_auc")
    gscv.fit(X_train.values, y_train)
    
    best_params = gscv.best_params_
    best_score = gscv.best_score_
    
    tuned_params = {k[12:]: v for k, v in best_params.items()}
    rf.set_params(**tuned_params)

    ######################
    # Step 2 - Fit the tuned Random Forest, determine rankings of features 
    ######################
    
    pipeline2 = PipelineRFE(steps=[('scaler', scaler), ('classifier', rf)])
    selector = RFECV(pipeline2, cv=5, step=1, scoring='roc_auc', verbose=0)
    selector.fit(X_train.values, y_train)
    cols = X_train.columns
    selected_cols = list(cols[selector.support_])
    
    rf.fit(X_train.values, y_train)
    feature_importances = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    
    ######################
    # Displayed output - number of columns selected, columns selected, auc graph, feature importance graph
    ######################
    
    print(); print()
    print(f'RFECV selected __ {len(selected_cols)} __ features out of {len(X_train.columns)} total features:')
    print(); print()
    print(selected_cols)
    print('\n\n')
    
    fig = plt.figure(figsize=(14, 5))
    fig.subplots_adjust(wspace=.4)
    
    # number of features vs. AUC line graph
    ax1 = fig.add_subplot(121)
    aucX = range(1, len(cols)+1)
    aucY = selector.cv_results_['mean_test_score']
    ax1.plot(aucX, aucY, '-bo', label='line with marker')
    ax1.xaxis.set_ticks(range(1, len(cols)+1, 2))
    ax1.set_xlabel('Number of Features Used')
    ax1.set_ylabel('AUC')
    ax1.set_title('Number of Best Features vs. AUC')
    ax1.grid(True)

    # feature importance bar graph
    ax2 = fig.add_subplot(122)
    ax2.bar(x=feature_importances.index, height=feature_importances.values)
    ax2.tick_params(axis='x', labelrotation = 270)
    ax2.set_title('Feature Importances')
    ax2.grid(False)

    plt.show()
    
    return feature_importances, selected_cols
    

def train_test_all_classifiers(X_train, X_test, y_train, y_test, plot_auc=False):

    ######################
    # Variables for training and testing
    ######################

    classifiers = {"LDA": LinearDiscriminantAnalysis(),
               "QDA": QuadraticDiscriminantAnalysis(),
               "AdaBoost": AdaBoostClassifier(),
               "Bagging": BaggingClassifier(),
               "Extra Trees Ensemble": ExtraTreesClassifier(),
               "Gradient Boosting": GradientBoostingClassifier(),
               "Random Forest": RandomForestClassifier(),
               "Ridge": RidgeClassifier(),
               "SGD": SGDClassifier(),
               "BNB": BernoulliNB(),
               "GNB": GaussianNB(),
               "KNN": KNeighborsClassifier(),
               "MLP": MLPClassifier(),
               "LSVC": LinearSVC(),
               "NuSVC": NuSVC(),
               "SVC": SVC(),
               "DTC": DecisionTreeClassifier(),
               "ETC": ExtraTreeClassifier()}
    decision_functions = ["Ridge", "SGD", "LSVC", "NuSVC", "SVC"]
    feature_importance = ["Gradient Boosting", "Extra Trees Ensemble", "Random Forest"]

    parameters = {}
    parameters.update({"LDA": {"classifier__solver": ["svd"]}})
    parameters.update({"QDA": {"classifier__reg_param":[0.01*ii for ii in range(0, 101)]}})
    parameters.update({"AdaBoost": {"classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                    "classifier__n_estimators": [200],
                                    "classifier__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]}})
    parameters.update({"Bagging": {"classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                   "classifier__n_estimators": [200],
                                   "classifier__max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                   "classifier__n_jobs": [-1]}})
    parameters.update({"Gradient Boosting": {"classifier__learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001], 
                                             "classifier__n_estimators": [200],
                                             "classifier__max_depth": [2,3,4,5,6],
                                             "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                             "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                             "classifier__max_features": ["auto", "sqrt", "log2"],
                                             "classifier__subsample": [0.8, 0.9, 1]}})
    parameters.update({"Extra Trees Ensemble": {"classifier__n_estimators": [200],
                                                "classifier__class_weight": [None, "balanced"],
                                                "classifier__max_features": ["auto", "sqrt", "log2"],
                                                "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                                "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                                "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                                "classifier__criterion" :["gini", "entropy"],
                                                "classifier__n_jobs": [-1]}})
    parameters.update({"Random Forest": {"classifier__n_estimators": [200],
                                         "classifier__class_weight": [None, "balanced"],
                                         "classifier__max_features": ["auto", "sqrt", "log2"],
                                         "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                         "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                         "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                         "classifier__criterion" :["gini", "entropy"],
                                         "classifier__n_jobs": [-1]}})
    parameters.update({"Ridge": {"classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]}})
    parameters.update({"SGD": {"classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                               "classifier__penalty": ["l1", "l2"],
                               "classifier__n_jobs": [-1]}})
    parameters.update({"BNB": {"classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]}})
    parameters.update({"GNB": {"classifier__var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5]}})
    parameters.update({"KNN": {"classifier__n_neighbors": list(range(1,31)),
                               "classifier__p": [1, 2, 3, 4, 5],
                               "classifier__leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                               "classifier__n_jobs": [-1]}})
    parameters.update({"MLP": {"classifier__hidden_layer_sizes": [(5), (10), (5,5), (10,10), (5,5,5), (10,10,10)],
                               "classifier__activation": ["identity", "logistic", "tanh", "relu"],
                               "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
                               "classifier__max_iter": [100, 200, 300, 500, 1000, 2000],
                               "classifier__alpha": list(10.0 ** -np.arange(1, 10))}})
    parameters.update({"LSVC": {"classifier__penalty": ["l2"],
                                "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]}})
    parameters.update({"NuSVC": {"classifier__nu": [0.25, 0.50, 0.75],
                                 "classifier__kernel": ["linear", "rbf", "poly"],
                                 "classifier__degree": [1,2,3,4,5,6]}})
    parameters.update({"SVC": {"classifier__kernel": ["linear", "rbf", "poly"],
                               "classifier__gamma": ["auto"],
                               "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100],
                               "classifier__degree": [1, 2, 3, 4, 5, 6]}})
    parameters.update({"DTC": {"classifier__criterion" :["gini", "entropy"],
                               "classifier__splitter": ["best", "random"],
                               "classifier__class_weight": [None, "balanced"],
                               "classifier__max_features": ["auto", "sqrt", "log2"],
                               "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                               "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                               "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10]}})
    parameters.update({"ETC": {"classifier__criterion" :["gini", "entropy"],
                               "classifier__splitter": ["best", "random"],
                               "classifier__class_weight": [None, "balanced"],
                               "classifier__max_features": ["auto", "sqrt", "log2"],
                               "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                               "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                               "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10]}})
    
    ######################
    # Now tune, train, and test every classifier
    ######################

    results = {}

    for clf_name, clf in classifiers.items():

        # scale, tune, and train
        print(f'Tuning {clf_name} . . . ')
        scaler = StandardScaler()
        pipe = Pipeline(steps=[('scaler', scaler), ('classifier', clf)])
        paramgrid = parameters[clf_name]
        gscv = GridSearchCV(pipe, param_grid=paramgrid, cv=5, n_jobs=-1, verbose=0, scoring='roc_auc')
        gscv.fit(X_train, y_train)

        # get best score and parameters for the classifier
        best_params = gscv.best_params_
        best_score = gscv.best_score_
        tuned_params = {k[12:]: v for k, v in best_params.items()}
        clf.set_params(**tuned_params)

        # use the best params to make predictions on test set
        # not all classifiers have same classification attributes!
        if clf_name in decision_functions:
            y_pred = gscv.decision_function(X_test)
        else:
            y_pred = gscv.predict_proba(X_test)[:,1]

        # evaluate our test set predictions
        auc = metrics.roc_auc_score(y_test, y_pred)

        # store the results for this classifier
        result = {'Classifier': gscv,
                 'Best Parameters': best_params,
                 'Training AUC': best_score,
                 'Test AUC': auc}
        results[clf_name] = result

    
    labels = list(classifiers.keys())
    train_auc = []
    test_auc = []

    for r in results:
        train_auc.append(results[r]['Training AUC'])
        test_auc.append(results[r]['Test AUC'])

    resdf = pd.DataFrame({'classifier': labels, 'train_auc': train_auc, 'test_auc': test_auc})
    
    if plot_auc:
        resdf.plot.bar(x='classifier', y=['train_auc', 'test_auc'], rot=90)
        plt.show()
    
    print(resdf.sort_values('test_auc', ascending=False)[['classifier', 'test_auc']][:5])
    
    return results, resdf.sort_values('test_auc', ascending=False)
    