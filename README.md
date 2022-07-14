# Telco churn classification project  
A school project to build a machine learning model to predict customer 'churn' and deliver a report on the indicators used in the model.

## About the project

The project will serve to demonstrate the Data Science (DS) Pipeline from end to end and will attempt to be as clear as possible about which steps are taken and when.  
As an overview, the DS Pipeline consists of the followings steps in order:
 - Planning
 - Acquisition
 - Preparation
 - Exploration
 - Preprocessing
 - Modeling
 - Delivery

### Steps to reproduce
First download a copy of each file or clone this repository.

To run this project you will need to fulfill one of the following requirements:
 > An environment file named `env.py` in the same directory as the other files.  The enviornment file should be structured the same as below and should contain your unique credentials to CodeUp's SQL server.
 ```py
hostname='data.codeup.com'
username='your_username'
password='your_password'
```
> A csv file named `telco.csv` in the same directory as the other files.  You can download a copy of the data from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)

You will also need to ensure the proper libraries are installed in your python environment.  You can install the libraries easily by running the following command in your python shell:
```py
%pip install numpy
%pip install pandas
%pip install matplotlib
%pip install seaborn
%pip install plotly
%pip install scipy
%pip install sklearn
```
___________

## Planning 
The planning phase of the pipeline will take place right here on this README.md file.  This is where we will define the goals of the project, and what it means to meet those goals.


Of course the primary goal of this project will be to build a model that can classify a customer as likely to churn by using other data-points about the customer to accurately predict the churn probability.  But what does it mean for a customer to "churn"?  What is the metric that we will use to determine how effective the model is at its predictions?  Why is it important to make these predictions in the first place?

These kinds of questions are important to ask during the planning phase, and neglecting them can lead to wasted effort and lost time from getting lost in rabbit holes and even potentially answering the wrong question.  A well written goal can answer many of these questions up-front.

### Goals

A telecom company is experiencing a high rate of customers discontinuing their service, an event that they refer to as "churn" or "turnover rate".  The company wants to know why their customers are churning so that they can patch any flaws in their service offerings.  They also want to be able to identify which customers are likely to churn in the future so that the customer retention team can pre-emptively contact them and attempt to retain them.

### Deliverables

Since the company wants two different things that means that we will need at least two deliverables in our final product:  
 1. A storytelling document that explains at least one driver of churn and makes recommendations on how to reduce churn.
 2. A machine learning model that predicts the probability of a customer churning that will perform well in production.

Because this is a classroom project there are some additional requirements that will need to be met.  

For the storytelling document:
 - It will need to be delivered in a jupyter notebook named `Final_report.ipynb`
 - It must include 4 visulalizations
 - It must include 2 statistical tests

For the machine learning model:
 - It should be demonstrated in the `Final_report.ipynb` notebook.

### Initial Hypothesis
Just like with any scientific endeavor, it is important to form a hypothesis when approaching a Data Science project.  The hypothesis gives us something concrete to start testing against.  To help us in choosing a good hypothesis let's take a look at what has happened so far in the scientific process.

| | | |
|---|---|---|
| Step 1 | Make an observation | The company has noticed a high rate of customer churn |
| Step 2 | Ask a question | What is causing customer churn |
| Step 3 | Form a Hypothesis | **We are here**|

How truthful the initial hypothesis turns out to be is ultimately unimportant since this is an iterive process and there will likely be many hypotheses throughout the life of the project.  The important part is to give us a starting point.  For the purposes of this project my initial hypothesis will be:
> Customers who pay the most are most likely to churn.

___
## Acquisition
The next step in the pipeline is to not only acquire the data that we will be working with, but also to pave a clear and repeatable path for bringing the raw data into our working environment.

The steps taken and necessary code for this process will first be tested in Jupyter Notebook `prep-explore.ipynb`, then they will be documented and compiled into callable functions stored in `wrangle.py` so that they can be reused in other artifacts created during the pipeline.

## Preparation
Once the data is in our environment the next thing to do is to start looking into the data to make sure it's useable for our project, correcting any issues you find, and making any transformations needed to feed the data into a machine learning model.

In addition we will create a data-dictionary that explains the features contained in the data.  We do this to familiarize ourselves with the data and to elimate any guess-work and incorrect assumptions that can occur when someone else looks at the data.  Here's ours:

### Data Dictionary
|Column | Description|
|---|---|
| `customer_id` | Unique identifier for the customer |
| `churn` | Yes/No - Yes if customer left company |
| `gender` | Gender of the customer |
| `senior_citizen` | 0/1 - 1 if the customer is a senior citizen |
| `partner` | Yes/No - Yes if customer has a spouse or partner |
| `dependents` | Yes/No - Yes if customer has children or dependents |
| `tenure` | Measure in months of how long customer has been with company |
| `phone_service` | Yes/No - Yes if customer has phone service |
| `multiple_lines` | Does customer with phone service have more than one line? |
| `online_security` | Does customer use online security services? |
| `online_backup` | Does customer use online backup services? |
| `device_protection` | Does customer use device protection services? |
| `tech_support` | Does customer use technical support services? |
| `streaming_tv` | Does customer use streaming television services? |
| `streaming_movies` | Does customer use movie streaming services? |
| `paperless_billing` | Does customer use paperless billing? |
| `monthly_charges` | Measure in USD that customer is billed monthly |
| `total_charges` | Measure in USD that customer has been charged since joining the company. |
| `internet_service_type` | The type of internet service customer uses. None if customer does not use internet. |
| `payment_type` | Customer's preferred payment method |
| `contract_type` | Renewal term for customer's contract. |

## Exploration

## Modeling

## Delivery