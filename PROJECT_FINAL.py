#!/usr/bin/env python
# coding: utf-8

# # Final assignment

# ## Alon Weisfeld - 308353994
# ## Idan Yacobi - 315310102

# In[265]:


import pandas as pd
import numpy as np
import re
import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# # Read Data

# In[266]:


data = pd.read_excel('C:\\Users\\alon3\\Downloads\\output_all_students_Train_v10.xlsx')


# # Drop NA price

# In[267]:


data = data.dropna(subset=['price'])


# # Columns as int

# ## Price 

# In[268]:


def num(x):
    if isinstance(x, str):
        x = re.sub(r'\D', '', x)
        return int(x) if x.isdigit() else 0
    elif isinstance(x, int):
        return x
    else:
        return 0
    
data['price']=data['price'].apply(lambda x: num(x)).replace('',np.nan)
data['price']=data['price'].astype(int)


# ## Area

# In[269]:


data['Area']=data['Area'].apply(lambda x: num(x)).replace('',np.nan)
data['Area']=data['Area'].astype(int)


# # Split Column floor

# In[270]:


def get_floor_and_totalfloor(data):
    def get_floor(x):
        if isinstance(x, str):
            x_list = x.split()
            if len(x_list) > 1:
                if x_list[1] == 'קרקע' or x_list[1] == 'מרתף':
                    return 0
                else:
                    return x_list[1]
            else:
                return 0
        else:
            return 0

    def get_totalfloor(x):
        if isinstance(x, str):
            x_list = x.split()
            if len(x_list) == 2:
                if x_list[1] == 'קרקע' or x_list[1] == 'מרתף':
                    return 0
                else:
                    return x_list[1]
            if len(x_list) == 4:
                return x_list[3]
            else:
                return 0
        else:
            return 0

    data['floor'] = data['floor_out_of'].apply(lambda x: get_floor(x)).astype('int64')
    data['total_floors'] = data['floor_out_of'].apply(lambda x: get_totalfloor(x)).astype('int64')
    return data

get_floor_and_totalfloor(data)
data = data.drop('floor_out_of', axis=1, inplace=False)


# # entrance_date column - categorical

# In[271]:


def entertime(x):
    category_mapping = {
        'גמיש': 'flexible',
        'לא צויין': 'not_defined',
        'מיידי': 'less_than_6_months'
    }
    
    if isinstance(x, str):
        x = x.strip()
        if x in category_mapping:
            return category_mapping[x]
    
    if isinstance(x, datetime.datetime):
        today = datetime.date.today()
        more6_months = today + datetime.timedelta(days=6 * 30)
        more12_months = today + datetime.timedelta(days=12 * 30)
        if x.date() < more6_months:
            return 'less_than_6_months'
        if more6_months <= x.date() <= more12_months:
            return 'months_6_12'
        if x.date() > more12_months:
            return 'above_year'
    
    return 'not_defined'

data['entranceDate ']=data['entranceDate '].apply(lambda x: entertime(x)).replace('',np.nan)


# # Boolean fields

# In[272]:


words_to_1 = ['כן', 'יש', 'נגיש', 'TRUE','yes']
words_to_0 = ['לא', 'אין', 'FALSE','no']
columns_to_boolean = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']

def convert_columns_to_boolean(data, columns_to_boolean, words_to_1, words_to_0):
    for column in columns_to_boolean:
        data[column] = data[column].astype(str)  
        for word in words_to_1:
            data[column] = np.where(data[column].str.contains(word, case=False, na=False), 1, data[column])
        for word in words_to_0:
            data[column] = np.where(data[column].str.contains(word, case=False, na=False), 0, data[column])
        data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0).astype(int)
    


convert_columns_to_boolean(data, columns_to_boolean, words_to_1, words_to_0)


# # Elastic-Net model 

# ## Data arrangement 

# In[274]:


data_test = pd.DataFrame(data, columns=['price','City','type','room_number','Area', 'city_area','Street', 'hasElevator ', 'hasParking ','hasBars ', 'hasStorage ', 'condition ','hasAirCondition ',
                              'hasBalcony ','hasMamad ','furniture ','entranceDate ', 'floor', 'total_floors'  ])

display(data_test)


# ### Clean room_number column 

# In[286]:


def room_num(x):
    if isinstance(x, str):
        x = re.sub(r'[^\d\.]', '', x)
        return x.replace(',', '')
    elif isinstance(x, int):
        return int(x) 
    else:
        return ''

    
data_test['room_number']=data_test['room_number'].apply(lambda x: room_num(x)).replace('',np.nan).astype('float64')


# In[287]:


data_test= data_test.dropna(axis=0, how='any', subset=None, inplace=False)
#בוצעה הורדה ולאחר בדיקה נשארו עם כ80% מהדאטה


# In[288]:


display(data_test)


# In[278]:


data_test['City']= data_test['City'] .str.replace('נהריה','נהרייה')


# In[279]:


data_test= pd.get_dummies(data_test, columns=['City','type', 'Street', 'city_area','condition ', 'furniture ', 'entranceDate '])


# In[280]:


display(data_test)


# ## Model

# In[283]:


X = data_test.drop('price', axis = 1)
y = data_test['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


alphas = [0.1, 0.5, 1.0, 5.0]  # List of alpha values to try
results = []  # Store results for each alpha

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for alpha in alphas:
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=0.5)
    elastic_net.fit(X_train_scaled, y_train)

    cross_val_rmse = np.sqrt(-cross_val_score(elastic_net, X, y, scoring='neg_mean_squared_error', cv=10))

    y_pred = elastic_net.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'alpha': alpha,
        'CV RMSE': cross_val_rmse.mean(),
        'Test MSE': mse,
        'Test MAE': mae,
        'Test R-squared': r2
    })

#Print performance metrics for each alpha
for result in results:
    print(f"Alpha: {result['alpha']}")
    print('Cross-validated RMSE:', result['CV RMSE'])
    print('Test set MSE:', result['Test MSE'])
    print('Test set MAE:', result['Test MAE'])
    print('Test set R-squared:', result['Test R-squared'])
    print()


# ## Plot

# In[282]:


alphas = [result['alpha'] for result in results]
test_mse = [result['Test MSE'] for result in results]

# Plot fot test set MSE- different alpha values
plt.plot(alphas, test_mse, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Test Set MSE')
plt.title('Test Set MSE for Different Alpha Values')
plt.xticks(alphas)
plt.grid(True)
plt.show()


# In[ ]:




