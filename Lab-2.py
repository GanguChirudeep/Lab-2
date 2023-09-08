#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
file_name = r"19CSE305_LabData_Set3.1.xlsx"
worksheet_name = 'thyroid0387_UCI'
df = pd.read_excel(file_name, sheet_name=worksheet_name)
df.replace("?", np.nan, inplace=True)
nominal=['sex', 'on thyroxine', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 
       'T3 measured', 'TT4 measured', 'T4U measured',
       'FTI measured', 'TBG measured', 'referral source',
       'Condition']
interval=['TSH','T3','TT4',  'T4U','FTI',  'TBG']
ratio=['age']

nominal_encoded = pd.get_dummies(df,columns=nominal)
df=nominal_encoded
df


# In[17]:


df.info()


# In[19]:


import numpy as np
df.describe()


# In[23]:


numeric_variables = df.select_dtypes(include=['int64', 'float64'])
data_range = numeric_variables.max() - numeric_variables.min()
print("Data Range for Numeric Variables:")
print(data_range)


# In[24]:


missing_values = df.isnull().sum()
print("Missing Values in Each Attribute:")
print(missing_values)


# In[25]:


#outliers for TSH
Q1 = df['TSH'].quantile(0.25)
Q3 = df['TSH'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_TT4 =df[(df['TSH'] < lower_bound) | (df['TSH'] > upper_bound)]
outliers_TT4


# In[26]:


#outliers for T3
Q1 = df['T3'].quantile(0.25)
Q3 = df['T3'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_TT4 =df[(df['T3'] < lower_bound) | (df['T3'] > upper_bound)]
outliers_TT4


# In[27]:


#outliers for TT4
Q1 = df['TT4'].quantile(0.25)
Q3 = df['TT4'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_TT4 =df[(df['TT4'] < lower_bound) | (df['TT4'] > upper_bound)]
outliers_TT4


# In[28]:


#outliers for T4U
Q1 = df['T4U'].quantile(0.25)
Q3 = df['T4U'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_TT4 =df[(df['T4U'] < lower_bound) | (df['T4U'] > upper_bound)]
outliers_TT4


# In[29]:


#outliers for FTI
Q1 = df['FTI'].quantile(0.25)
Q3 = df['FTI'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_TT4 =df[(df['FTI'] < lower_bound) | (df['FTI'] > upper_bound)]
outliers_TT4


# In[30]:


#outliers for TBG
Q1 = df['TBG'].quantile(0.25)
Q3 = df['TBG'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_TT4 =df[(df['TBG'] < lower_bound) | (df['TBG'] > upper_bound)]
outliers_TT4


# In[31]:


numeric_variables = df.select_dtypes(include=['int64', 'float64'])
numeric_mean = numeric_variables.mean()
numeric_variance = numeric_variables.var() 
numeric_std = numeric_variables.std()
print("Mean of Numeric Variables:")
print(numeric_mean)
print("\nVariance of Numeric Variables:")
print(numeric_variance)
print("\nStandard Deviation of Numeric Variables:")
print(numeric_std)


# In[32]:


#'TSH','T3','TT4',  'T4U','FTI','TBG' are numerical data with outliers, so we replace missing values with median
df['TSH'].fillna(df['TSH'].median(), inplace=True)
df['T3'].fillna(df['T3'].median(), inplace=True)
df['TT4'].fillna(df['TT4'].median(), inplace=True)
df['T4U'].fillna(df['T4U'].median(), inplace=True)
df['FTI'].fillna(df['FTI'].median(), inplace=True)
df['TBG'].fillna(df['TBG'].median(), inplace=True)
df


# In[33]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale = ['age','TSH','T3','TT4','T4U','FTI','TBG']
df[scale] = scaler.fit_transform(df[scale])
df


# In[34]:


v1 = df['sex_M']
v2 = df['Condition_M']
f11 = sum([1 for a, b in zip(v1, v2) if a == b == 1])
f01 = sum([1 for a, b in zip(v1, v2) if a == 0 and b == 1])
f10 = sum([1 for a, b in zip(v1, v2) if a == 1 and b == 0])
f00 = sum([1 for a, b in zip(v1, v2) if a == b == 0])
print(f00,f01,f10,f11)


# In[35]:


jc = f11 / (f01 + f10 + f11)
jc


# In[36]:


smc = (f11 + f00) / (f00 + f01 + f10 + f11)
smc


# In[37]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

v1 = np.array(v1).reshape(1, -1)
v2 = np.array(v2).reshape(1, -1)

cosine_sim = cosine_similarity(v1, v2)
print("Cosine Similarity:", cosine_sim[0][0])


# In[39]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define a list of vectors (replace with your data)
vectors = [
    [1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 1],
    # Add more vectors here
]

# Function to calculate Jaccard similarity
def jaccard_similarity(vector1, vector2):
    intersection = len(set(vector1) & set(vector2))
    union = len(set(vector1) | set(vector2))
    return intersection / union

# Function to calculate Simple Matching Coefficient (SMC) similarity
def smc_similarity(vector1, vector2):
    matching_elements = np.sum(np.logical_and(vector1, vector2))
    total_elements = np.sum(np.logical_or(vector1, vector2))
    return matching_elements / total_elements

# Function to calculate Cosine similarity
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

# Calculate similarity matrices
num_vectors = len(vectors)
jc_similarity_matrix = np.zeros((num_vectors, num_vectors))
smc_similarity_matrix = np.zeros((num_vectors, num_vectors))
cosine_similarity_matrix = np.zeros((num_vectors, num_vectors))

for i in range(num_vectors):
    for j in range(num_vectors):
        jc_similarity_matrix[i, j] = jaccard_similarity(vectors[i], vectors[j])
        smc_similarity_matrix[i, j] = smc_similarity(vectors[i], vectors[j])
        cosine_similarity_matrix[i, j] = cosine_similarity(vectors[i], vectors[j])

# Create subplots for each similarity matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot Jaccard similarity heatmap
sns.heatmap(jc_similarity_matrix, annot=True, ax=axes[0], cmap="YlGnBu", xticklabels=False, yticklabels=False)
axes[0].set_title("Jaccard Similarity")

# Plot Simple Matching Coefficient (SMC) similarity heatmap
sns.heatmap(smc_similarity_matrix, annot=True, ax=axes[1], cmap="YlGnBu", xticklabels=False, yticklabels=False)
axes[1].set_title("SMC Similarity")

# Plot Cosine similarity heatmap
sns.heatmap(cosine_similarity_matrix, annot=True, ax=axes[2], cmap="YlGnBu", xticklabels=False, yticklabels=False)
axes[2].set_title("Cosine Similarity")

plt.show()


# In[ ]:




