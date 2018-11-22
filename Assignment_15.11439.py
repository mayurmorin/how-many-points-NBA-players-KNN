
# coding: utf-8

# ## In this assignment, I will be using the K-nearest neighbors algorithm to predict how many points NBA players scored in the 2013-2014 season.

# <p>Before we dive into the algorithm, let’s take a look at our data. Each row in the data contains information on how a player performed in the 2013-2014 NBA season. </p>
# <p>Download 'nba_2013.csv' file from this link: https://www.dropbox.com/s/b3nv38jjo5dxcl6/nba_2013.csv?dl=0 </p>
# <p>Here are some selected columns from the data:</p>
# <ul><li>player - name of the player</li>
# <li>pos - the position of the player</li>
# <li>g - number of games the player was in</li>
# <li>gs - number of games the player started</li>
# <li>pts - total points the player scored</li></ul>
# <p>There are many more columns in the data, mostly containing information about average player game performance over the course of the season. See this site for an explanation
# of the rest of them.</p>

# ## Importing Modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading Data

# In[2]:


#Reading data from downloaded CSV file.
with open("nba_2013.csv", 'r') as csvfile:
    nba = pd.read_csv(csvfile)


# ## Data Exploration

# In[3]:


nba.columns.values #The names of all the columns in nba dataframe.


# In[4]:


nba.shape #Return a tuple representing the dimensionality of nba DataFrame.


# In[5]:


nba.head() #Returns the first 5 rows of nba dataframe.


# In[6]:


nba.info() #Prints information about nba dataframe.


# In[7]:


nba.describe() #The summary statistics of the nba dataframe


# In[8]:


nba.isnull().values.any() #Check for any NA’s in the dataframe.


# In[9]:


#Shows percentage of data column wise missing in nba dataframe.
total = nba.isnull().sum().sort_values(ascending=False)
percent_1 = nba.isnull().sum()/nba.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data


# <p>There are missing data in x3p. ,ft. ,x2p. ,fg. and efg.	columns.</p>

# ## Data Visualization

# In[10]:


#Scatter Plot of "number of games the player was in" and "total points the player scored"
plt.figure(figsize=(16,9))
plt.title("Scatter Plot")
plt.xlabel("Number of games the player was in")
plt.ylabel("Total points the player scored")
plt.scatter(nba['g'],nba['pts'])
plt.show()


# In[11]:


#Histogram Plot of Number of games
plt.figure(figsize=(16,9))
plt.xlabel("Number of games")
plt.hist(nba['g'],bins=30)


# In[12]:


#Histogram Plot of Total points the player scored
plt.figure(figsize=(16,9))
plt.xlabel("Total points the player scored")
plt.hist(nba['pts'],bins=30)


# ## Data Imputation

# In[13]:


#Replacing null values with '0'
nba_new = nba.fillna(0)


# In[14]:


nba_new.isnull().values.any() #Check for any NA’s in the dataframe.


# In[15]:


#Dropping player, bref_team_id and season columns
nba_new.drop(["player","bref_team_id", "season"], axis=1, inplace=True)


# In[16]:


#Converts categorical column 'pos'  data into dummy variables
nba_df = pd.get_dummies(data=nba_new,columns=["pos"])


# ### Train, Test & Split

# In[17]:


#Selecting features and target
X=nba_df.drop("pts",axis=1)
y=nba_df.pts


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
print(X_train.shape) #Training data shape (predictor values) : 70%
print(X_test.shape) #Test data shape (predictor values) : 30%
print(y_train.shape) #Training data shape (target values) : 70%
print(y_test.shape) #Test data shape (target values) : 30%


# ### Creating and Training the Model

# In[19]:


#Instantiating learning model (k = 5)
knn = KNeighborsRegressor(n_neighbors=5)

#Fitting the model
knn.fit(X_train, y_train)


# ### Predicting "total points the player scored" using Test Data

# In[20]:


#Predicting "total points the player scored" using test data set
pred = knn.predict(X_test)


# In[21]:


#Prints first five predicted "total points the player scored" values 
print(pred[:5])


# In[22]:


#Scatter Plot of "Actual total points the player scored" and "Predicted total points the player scored"
plt.figure(figsize=(16,9))
plt.title("Scatter Plot")
plt.xlabel("Actual total points the player scored")
plt.ylabel("Predicted total points the player scored")
plt.scatter(y_test,pred)
plt.show()


# ### Evaluating the model

# In[23]:


#Caluculating and printing RMSE value for the model
print(np.sqrt(mean_squared_error(y_test,pred)))

