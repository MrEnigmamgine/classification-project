# Common DS imports
import pandas as pd
import numpy as np
from scipy import stats

# Custom helper library
import wrangle

# SKlearn Libraries
from sklearn.linear_model import LogisticRegression

seed = 8

def get_best_model():
    # Import the ready to model data
    df = wrangle.get_tidier_telco_data()
    # Split off the training data
    train, test, validate, verify = wrangle.train_test_validate_verify_split(df)
    # Separate the features from the target
    X_train, y_train = wrangle.x_y_split(train)
    # Create the algorithm class
    model = LogisticRegression(max_iter=200, random_state=8)
    # and fit it to our training data
    model.fit(X_train,y_train) 
    
    return model


def chi2_test(df, alpha=0.05):
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