import pandas as pd
import os

def get_db_url(database, hostname='', username='', password='', env=''):
    '''Creates a URL for a specific database and credential set to be used with pymysql.

    Can be used either with a set of credentials passed directly to the function or with an environment file containing the credentials.
    If both are provided, the environment file takes precedence.

    Parameters:
    database (str): The target database that pymysql will connect to, which will provide context for any SQL queries used in the connection.
    hostname (str): The DNS hostname or IP-Adress for the connection
    username (str), password (str): User credentials that will be used in a sql connection.
    env (str): Relative path to an environment file.  The file must include the hostname, username, and password variables.

    Returns:
    str: Full URL for use with a pymysql connection
    '''
    if env != '':
        d = {}
        file = open(env)
        for line in file:
            (key, value) = line.split('=')
            d[key] = value.replace('\n', '').replace("'",'').replace('"','')
        username = d['username']
        hostname = d['hostname']
        password = d['password']
    url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return url

def new_telco_data():
    url = get_db_url('telco_churn',env='./env.py')
    query = """
        SELECT 
            *
        FROM
            telco_churn.customers
                JOIN
            telco_churn.internet_service_types USING (internet_service_type_id)
                JOIN
            telco_churn.payment_types USING (payment_type_id)
                JOIN
            telco_churn.contract_types USING (contract_type_id)
        ;
        """
    df = pd.read_sql(query, url)
    return df

def get_telco_data():
    filename = "telco.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_telco_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  


def get_tidy_telco_data():
    # Get a fresh copy of the data
    df = get_telco_data()
    # Drop columns that just repeat the data in other columns
    df = df.drop(columns=['contract_type_id', 'payment_type_id', 'internet_service_type_id'])
    # Replace the spaces in total_charges with a 0 and sets the column's type to a float
    df.total_charges = df.total_charges.replace(' ', 0).astype(np.float64)
    # Convert the senior citizen column into boolean type.
    # We want to handle this one separetely because unlike the others, it uses 1's and 0's instead of Yes/No
    df.senior_citizen = df.senior_citizen == 1
    # Do something similar with gender
    df.gender = df.gender == 'Male'
    # Also rename the gender column to make it more intuitive to read later.
    df.rename(columns={'gender': 'is_male'}, inplace=True)
    # Loop through the yes/no columns and convert them to booleans.
    yes_no_columns = [ 'partner',
                    'dependents',
                    'phone_service',
                    'multiple_lines',
                    'online_security',
                    'online_backup',
                    'device_protection',
                    'tech_support',
                    'streaming_tv',
                    'streaming_movies',
                    'paperless_billing',
                    'churn']
    for column in yes_no_columns:
        df[column] = df[column] == "Yes"
    return df

def get_tidier_telco_data():
    df = get_tidy_telco_data()
    return pd.get_dummies(df, columns=['internet_service_type', 'payment_type', 'contract_type'], dtype=bool)