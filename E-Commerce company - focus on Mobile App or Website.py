#!/usr/bin/env python
# coding: utf-8

# # Context
# 
# I have been awarded contract work by an Ecommerce company based in New York City that sells clothing online but also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired me on contract to help them figure it out! 
# 
# ## Dataset
# 
# [Ecommerce Customers](https://github.com/Sharma-Amol/E-Commerce_company-Focus_on_mobile_app_or_website/blob/main/Ecommerce%20Customers.csv)
# 
# ### Note
# 
# All personal information like emails and address are counterfeit data.

# # Let's begin.

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Getting the Data
# 
# I'll work with the 'Ecommerce Customers' csv file provided by the company. It has Customer info, such as Emails, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 

# In[2]:


customers = pd.read_csv('Ecommerce Customers.csv')


# In[3]:


# Checking first 5 rows to know the type of data I have.

customers.head()


# In[4]:


# A quick statistical summary of numerical column values.

customers.describe()


# In[5]:


# TO know the data types of all column values.

customers.info()


# ## Exploratory Data Analysis
# 
# FOr this exercise I'll be using only the numerical data of the csv file.

# In[6]:


# To know labels of all columns.

customers.columns


# In[7]:


# To know labels of all rows

customers.index


# In[8]:


# Using seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. 
# To check if the correlation make sense.

sns.jointplot(x ='Time on Website',y='Yearly Amount Spent',data=customers)


# Can see that more time spent on website means more money spent.

# In[9]:


# Doing the same but with the Time on App column instead.

sns.jointplot(x ='Time on App',y='Yearly Amount Spent',data=customers)


# Visually, we can see there is some correlation between time on app and yearly amount spent.

# In[10]:


# Using jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.

sns.jointplot(x ='Time on App',y='Length of Membership',data=customers,kind='hex')


# In[11]:


# I'll now explore these types of relationships across the entire data set. 
# Using pairplot to know the correlations.

sns.pairplot(customers)


# Based on this plot, the most correlated feature appears to be between length of membership and yearly amount spent.

# In[12]:


# Creating a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# The longer you stay a member, the higher is the yearly amount spent

# In[13]:


customers.columns


# ## Training and Testing Data
# 
# After Exploratory Data Analysis, I'll now split the data into training and testing sets.

# In[14]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership',]]
y = customers['Yearly Amount Spent']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model

# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lm = LinearRegression()


# In[19]:


lm.fit(X_train,y_train)


# **Printing out the coefficients of the model**

# In[20]:


lm.coef_


# ## Predicting Test Data
# Model is now fit. Let's evaluate its performance by predicting off the test values!

# In[21]:


predictions = lm.predict(X_test)


# In[22]:


# Creating a scatterplot of the real test values versus the predicted values. 

sns.scatterplot(x=y_test,y=predictions)


# ## Evaluating the Model
# 
# I'll now evaluate the model performance by calculating the residual sum of squares and the explained variance score (R^2).

# In[23]:


from sklearn import metrics


# In[24]:


metrics.mean_absolute_error(y_test,predictions)


# In[25]:


metrics.mean_squared_error(y_test,predictions)


# In[26]:


np.sqrt(metrics.mean_squared_error(y_test,predictions)) #RMSE


# In[27]:


# variance score (R^2). It tells about how much variance our model explains.
metrics.explained_variance_score(y_test,predictions)


# ## Residuals
# 
# As the data is counterfeit, the model has very good fit. I'll quickly explore the residuals to make sure everything was okay with the data. 

# In[28]:


sns.set(style='darkgrid')
sns.displot(y_test-predictions,bins=50,kde=True)


# ## Conclusion
# 
# Now to original question, should the company focus their efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's interpret the coefficients to get an idea.

# In[29]:


cdf = pd.DataFrame(data=lm.coef_,index=X_train.columns,columns=['Coeff.'])
cdf


# How to interpret these coefficients? 

# If you hold all the other features fixed, 
# 
# a.) A one unit increase in Avg. session Length is associated with $25.98 more expense.
# 
# b.) A one unit increase in Time on App is associated with $38.59 more expense.
# 
# c.) A one unit increase in Time on Website is associated with $0.19 more expense.
# 
# d.) A one unit increase in Length of Membership is associated with $61.27 more expense.
# 
# So, mobile app experience or website?
# 
# There are two ways to look at this question.
# 
# One way to go is that the website needs more work to catch up to mobile app.
# 
# Another way is to develop app more since it is already working much better.
# 
# I need more data on connection between Length of membership and app and website to come to a binary decision.
# 
# Additional factors like how much is it going to cost to upgrade the website in comparison to app, other company specific features are also to be taken into consideration.
# 
# Nonetheless, I have a fully informed background to come to my conclusion.
