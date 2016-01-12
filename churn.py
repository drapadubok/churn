import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import colorlover as cl
## Credentials for plotly, feel in your own!
py.sign_in('uname', 'apikey')


def load_churn_data():
    """
    Load dataset from http://www.dataminingconsultant.com/DKD.htm book.
    
    Output:
        pandas DataFrame
    """
    data = pd.read_csv('https://s3.eu-central-1.amazonaws.com/cv-ds-portfolio/churn.csv')
    return data
    

def clean_churn_data(data):
    """
    Drops some irrelevant columns, turns "Churn?" into y, and else into X.
     
    Input: 
        churn dataset from load_churn_data()
     
    Output:
        y - labels
        X - features
    """
    y = np.where(data['Churn?']=='True.',1,0)
    
    # drop labels and irrelevant columns
    data_no_y = data.drop(['Phone','Area Code','State','Churn?'],axis=1)
    
    # Some columns are yes/no, turn them into boolean
    yes_no_cols = ["Int'l Plan","VMail Plan"]
    data_no_y[yes_no_cols] = data_no_y[yes_no_cols] == 'yes'
    
    # Pull out feature names for future use
    feature_names = data_no_y.columns
    
    # Convert to matrix
    X = data_no_y.as_matrix().astype(np.float)
    
    # zscore
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, feature_names
    

def wrap_estimator(estimator, X, y, cv):
    """
    Wrapper for classification, good to try different estimators
    
    Input:
        estimator - classifier from sklearn toolbox
        X - features
        y - labels
        cv - cross-validation scheme
        
    Output:    
        yhat - 
        y_prob - 
        cv_scores -        
    """
    # Init containers for output
    cv_scores = []
    yhat = y.copy()
    y_prob = np.zeros((len(y),2))
    
    for train, test in cv:
        estimator.fit(X[train,:], y[train])
        yhat[test] = estimator.predict(X[test,:])
        
        # Check if has predict_proba method, get probs if yes
        proba = getattr(estimator, "predict_proba", None)
        if callable(proba):
            y_prob[test] = estimator.predict_proba(X[test,:])
        else:
            y_prob = None
        
        cv_scores.append(np.sum(yhat[test] == y[test]) / float(np.size(y[test])))
    return yhat, y_prob, cv_scores
    
   
def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    From http://planspace.org/20151129-see_sklearn_trees_with_d3/
    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {}'.format(feature, threshold)
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node
    

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    ''' Parameters:
    cm - confusion matrix,
    title - title
    cmap - colormap, default plt.cm.Blues '''
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    img = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(img,ax=ax)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off',  # labels along the bottom edge are off
        labelleft='off',
        right='off')
    return fig
    
    
def get_10_worst(preds):
    """
    Given probabilities of churn outcome, returns the indices of top 10 probs
    
    Input:
        preds - probability of Churn outome for each client, or
        second column of predict_proba method of sklearn models
        
    Output:
        top10churn - index and probability of churning 
    """
    proba_index = np.argsort(preds)
    last10 = proba_index[-10:]
    proba_val = preds[last10]
    # format nicely    
    top10churn = pd.DataFrame(last10,proba_val).reset_index()
    top10churn.columns = ['pred_prob','client_index']
    
    return top10churn
    

# Prep the data
data = load_churn_data()
X,y,feature_names = clean_churn_data(data)
# Cross validation scheme
cv = KFold(n=len(y), n_folds=5, shuffle=True)
# Classification
clf = RandomForestClassifier(n_estimators=100)
yhat, y_prob, scores = wrap_estimator(clf, X, y, cv)

# Results
accuracy = np.mean(scores)
cm = confusion_matrix(y,yhat)
# Get importance of features
clf.fit(X,y)
importances = clf.feature_importances_
# Scale importance
importances = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(importances)
feature_names_sorted = [feature_names[i] for i in sorted_idx]
importances.sort()
# ROC curve
preds = y_prob[:,1]
fpr, tpr, _ = metrics.roc_curve(y, preds)
roc_auc = metrics.auc(fpr, tpr)
#
top10churn = get_10_worst(preds)


##### Plotting section #####
data = go.Data([
    go.Scatter(
        x=fpr,
        y=tpr,
        name='ROC Curve'
    )
])
layout = go.Layout(
    title='ROC Curve',
    xaxis=go.XAxis(
        title='False positive rate'
    ),
    yaxis=go.YAxis(
        title='True positive rate',
        type='linear'
    )
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename="ROC")

# Confusion matrix
colormap = cl.scales['9']['seq']['Reds']
colorrange = np.linspace(1,len(y),9,dtype=int)
clscale = [list(a) for a in zip(colorrange,colormap)]
data = [
    go.Heatmap(
        z=[cm[1], cm[0]],
        x=['Not Churned', 'Churned'],
        y=['Churned', 'Not Churned'],
        autocolorscale=False,
        colorscale=clscale
    )
]
layout = go.Layout(
    barmode='overlay',
    title='Confusion Matrix',
    xaxis=go.XAxis(
        title='Predicted value',
        titlefont=dict(
            color='#7f7f7f',
            size=18
        )
    ),
    yaxis=go.YAxis(
        title='True Value',
        titlefont=dict(
            color='#7f7f7f',
            size=18
        )
    ),
    width=400,
    height=400
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename="confmat_rdy")

# Barplot
data = [
    go.Bar(
        x=feature_names_sorted,
        y=importances
    )
]
layout = go.Layout(
    title='Feature importances',
    xaxis=go.XAxis(
        title='Scaled importance',
        titlefont=dict(
            color='#7f7f7f',
            size=18
        )
    ),
    yaxis=go.YAxis(
        title='Feature label',
        titlefont=dict(
            color='#7f7f7f',
            size=18
        )
    )
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename="barplot")