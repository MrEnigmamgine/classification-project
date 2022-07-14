# Common DS imports
import pandas as pd
import numpy as np
from scipy import stats

# Custom helper library
import wrangle

# SKlearn Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

import plotly.express as px

seed = 8

def get_model():
    """Returns a model trained on an upsampled training set.  
    The model's hyperparameters are tuned for the optimum performance found during testing."""
    # Import the ready to model data
    df = wrangle.get_tidier_telco_data()

    # Split off the training data
    train, test, validate, verify = wrangle.train_test_validate_verify_split(df)

    # Upsample the training data to balance a class imbalance
    minority_upsample = resample( train[train.churn],   #DF of samples to replicate
                                replace = True,         #Implements resampling with replacement, Default=True
                                n_samples = len(train[~train.churn]), #Number of samples to produce
                                random_state= 8         #Random State seed for reproducibility
                                )
    #Then glue the upsample to the original
    train = pd.concat([minority_upsample, train[~train.churn]])

    # Separate the features from the target
    X_train, y_train = wrangle.x_y_split(train)

    # Create the algorithm class
    model = LogisticRegression(max_iter=200, random_state=8)

    # and fit it to our training data
    model.fit(X_train,y_train) 
    
    return model


def chi2_test(df, alpha=0.05):
    """Performs a chi2 test on a crosstap-like dataframe.
    Intended to declutter the final report notebook."""
    chi2, p, degf, expected = stats.chi2_contingency(df)
    print('Observed\n')
    print(df.values)
    print('---\nExpected\n')
    print(expected.astype(int))
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'degf = {degf}')
    print(f'p     = {p:.4f}')
    print('---\n')
    if p < alpha:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")

def confusion_matrix_treemap(tn,fp,fn,tp):
    """Draws a treemap from confusion matrix data.
    This method was clunky, unwieldy, and used a lot of hard-coding, so I thought it was best to hide it in the helper library."""
    fig = px.treemap(   title='Model Predictions compared to truth',
                        values= [0,0]+[tn,fp,fn,tp] ,
                        names= ['churned','not churned','true_negative','false_positive','false_negative','true_positive'],
                        parents=['','', 'not churned','not churned','churned','churned'],
                        width=600, height=400,
                        color=['churned','not churned','true_negative','false_positive','false_negative','true_positive'],
                        color_discrete_map= {'churned':'FF221F',
                                            'not churned':'099C4E',
                                            'true_negative':'099C4E',
                                            'false_positive':'FF221F',
                                            'false_negative':'099C4E',
                                            'true_positive':'FF221F'}
                        )

    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin = dict(t=45, l=0, r=0, b=0))
    fig.show()