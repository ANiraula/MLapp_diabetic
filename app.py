#End-to-End ML
#by Anil Niraula

#Steps----------
#[1] Data loading
#[2] Data cleaning
#[3] Train/test split
#[4] Feature preprocessing (Noramlize/Standardize)
#[5] Feature selection
#[6] Train/fit models


# In[135]:


import pandas as pd
#[1] Data

#SOURCE: https://data-usdot.opendata.arcgis.com/datasets/alternative-fueling-stations/explore
data = pd.read_csv("https://raw.githubusercontent.com/ANiraula/MLapp_diabetic/main/NHANES_age_prediction.csv")
data = pd.DataFrame(data)
data = data.loc[data['DIQ010'] < 3]
data.loc[data['DIQ010'] == 3]
#data.head()


# In[136]:


#Exploratory Data Analysis
#print(data.groupby('DIQ010')['age_group','BMXBMI'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')


# In[137]:


#[2] Data cleaning
data['age_group']= data['age_group'].astype('category')
data['age_group'] = data['age_group'].cat.codes

# fillNA
data2 = data
#Define which values replace NA with for which columns
values = {'SEQN':0, 'age_group':0, 'RIDAGEYR':0, 'RIAGENDR':0, 'PAQ605':0,
                    'BMXBMI':0, 'LBXGLU':0, 'DIQ010':0, 'LBXGLT':0, 'LBXIN':0}

data2.fillna(value=values)
#data2.info()


# In[173]:


#[3] Tarin/test split

from sklearn.model_selection import train_test_split
data_y = data2['DIQ010']
data_x = data2.drop(['DIQ010'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,test_size = 0.2, random_state=55)
#print(X_train['age_group'].value_counts().sum())
#print(X_test['age_group'].value_counts().sum())
#print(data['age_group'].value_counts().sum())


# In[158]:


X_train = X_train[['RIDAGEYR', 'LBXGLU','BMXBMI']]
X_test = X_test[['RIDAGEYR', 'LBXGLU','BMXBMI']]


# In[192]:


#[4] Noramlize/Standardize

from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
X_train_norm = norm.fit_transform(X_train.values.reshape(-1,1))

# Use the same normalizer to transform the 'age' column of the test set to avoid data leakage
X_test_norm = norm.transform(X_test.values.reshape(-1,1))

y_train_norm = norm.fit_transform(y_train.values.reshape(-1,1))
y_test_norm = norm.transform(y_test.values.reshape(-1,1))

#print(len(X_train_norm))
#print(len(y_train_norm))
#print(len(X_test_norm))
#print(len(y_test_norm))


#[6] Train model
from sklearn.linear_model import LogisticRegression
import numpy as np

#Create model
logistic_model = LogisticRegression(max_iter=300, class_weight = 'balanced')
#Train
logistic_model.fit(X_train , np.ravel(y_train,order='C'))

#Create dummy input (age, clucose & BMI)
pred_dummy = X_train[0:1]*1.5
#print(pred_dummy)

#Prediction probabilities
pred_dummy_prob = logistic_model.predict_proba(pred_dummy)

#Prediction
pred_dummy_predict = logistic_model.predict(pred_dummy)
#print(pred_dummy)
#print('Prob Diabetic vs. non-Diabetic', pred_dummy_prob)
#print('2. Non-Diabetic: ', pred_dummy_predict) if pred_dummy_predict == 2 else print('1. Diabetic: ', pred_dummy_predict)
#print(pred_dummy['RIDAGEYR'])


# In[199]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output

#app = dash.Dash(__name__)
#server = app.server

pred_dummy = X_train[0:1]*2
app.run_server(mode="inline", host="localhost",port=8054)
# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
        html.Label('Age'),
    dcc.Input(
        id='Age',
        type='number',
        value=35,
        style={'marginRight': '10px'}
    ),
        html.Label('Gluten'),
    dcc.Input(
        id='Gluten',
        type='number',
        value=120,
        style={'marginRight': '10px'}
    ),
        html.Label('BMI'),
    
    dcc.Input(
        id='BMI',
        type='number',
        value=25,
        style={'marginRight': '10px'}
    ),
    html.Button('Predict', id='add-button', n_clicks=0),
    html.Div(id='output')
])

# Define the callback to update the output div
@app.callback(
    Output(component_id='output', component_property='children'),
    [Input(component_id='Age', component_property='value'),
     Input(component_id='Gluten', component_property='value'),
     Input(component_id='BMI', component_property='value'),
     Input(component_id='add-button', component_property='n_clicks')]
)
def update_output_div(input1, input2, input3, n_clicks):
    if n_clicks > 0:
        pred_dummy['RIDAGEYR'] = input1
        pred_dummy['LBXGLU'] = input2
        pred_dummy['BMXBMI'] = input3
        pred_dummy_prob = logistic_model.predict_proba(pred_dummy)
        pred_dummy_predict = logistic_model.predict(pred_dummy)
        
        result =  pred_dummy_prob[[0]].round(3)*100
        result2 = 'Not Diabetic' if pred_dummy_predict == 2 else 'Diabetic'


        return f'* 'f' {result2}'f' *  'f' |  Prob Diabetic vs. non-Diabetic (%) : {result}' 
    return ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
