# --------------------------------------------Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------Figures inline and set visualization style
sns.set()

# --------------------------------------------Read CSV with pandas----------------------------
df = pd.read_csv(r'advertising-1.csv')

# -------------------------------------------- EDA --------------------------------------------
# -------------------------------------------- Most of the internet users are having age in the range of 26 to 46 years ------
sns.distplot(df['Age'], bins=10, kde=True,
             hist_kws=dict(edgecolor="k", linewidth=2))
sns.distplot(df['Age'], hist=False, color='r', rug=True, fit=norm)

print('\nMax Age:', df['Age'].max(), 'Years')
print('Min Age:', df['Age'].min(), 'Years')
print('Average Age:', df['Age'].mean(), 'Years')

# -------------------------------------------- Income distribution w.r.t age --------------------------
sns.jointplot(x='Age', y='Area Income', data=df)

# --------------------------------------------Age group  spending max time on the internet---------------------------
# More time is spent by young users.

sns.jointplot(x='Age', y='Daily Time Spent on Site', data=df)
sns.catplot(x='Age', kind='box', data=df)

# --------------------------------------------Gender with more clicks on online ads--------------
print('\nGender with more clicks on online ads \n', df.groupby(
    ['Male', 'Clicked on Ad'])['Clicked on Ad'].count().unstack())

# --------------------------------------------User base from each coutry(sorted list high to low)----------
print('\n Maximum user base from coutry\n', pd.crosstab(
    index=df['Country'], columns='count').sort_values(['count'], ascending=False))

# --------------------------------------------Mean of Daily Time Spent on Site	Age	Area Income	Daily Internet Usage ----------
print('\n Mean: \n', df.groupby('Clicked on Ad')['Clicked on Ad', 'Daily Time Spent on Site', 'Age', 'Area Income',
                                                 'Daily Internet Usage'].mean())

# --------------------------------------------Relationship analysis: Correlation on Numeric data set----------
correlation = df.corr()
print('\nCorrelation: \n', correlation)

sns.heatmap(correlation, xticklabels=correlation.columns,
            yticklabels=correlation.columns, annot=True)
sns.pairplot(df, hue='Clicked on Ad')

# -------------------------------------------- Relational Plots  -------------
sns.relplot(x='Male', y='Daily Time Spent on Site',
            hue='Clicked on Ad', data=df)
sns.relplot(x='Area Income', y='Daily Time Spent on Site',
            hue='Clicked on Ad', data=df)
sns.relplot(x='Age', y='Daily Time Spent on Site',
            hue='Clicked on Ad', data=df)
sns.relplot(x='Daily Internet Usage', y='Daily Time Spent on Site',
            hue='Clicked on Ad', data=df)

# --------------------------------------------Categorical plot-------------
sns.catplot(x='Area Income', kind='box', data=df)
sns.catplot(x='Daily Time Spent on Site', kind='box', data=df)
sns.catplot(x='Daily Internet Usage', kind='box', data=df)
# plt.show()


# --------------------------------------------DATA CLEANING--------------------------------------------

# -------------------------------------------- Setting the data types--------------------------

# set categorical data
df['Male'] = df['Male'].astype('category')
df['Clicked on Ad'] = df['Clicked on Ad'].astype('category')
df['City Codes'] = df['City'].astype('category').cat.codes
df['Country Codes'] = df['Country'].astype('category').cat.codes
df['Ad Topic Line'] = df['Ad Topic Line'].astype('category')

# set numeric data
df['Daily Time Spent on Site'] = pd.to_numeric(
    df['Daily Time Spent on Site'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Area Income'] = pd.to_numeric(df['Area Income'], errors='coerce')
df['Daily Internet Usage'] = pd.to_numeric(
    df['Daily Internet Usage'], errors='coerce')

# Add Date/Time colums from timestamp column
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Month'] = df['Timestamp'].dt.month
df['Day of the month'] = df['Timestamp'].dt.day
df["Day of the week"] = df['Timestamp'].dt.dayofweek
df['Hour'] = df['Timestamp'].dt.hour
df = df.drop(['Timestamp'], axis=1)

# --------------------------------------------Print Unique values-------------------------
print('\n Unique Values')
print(df.nunique())
# print(df['Age'].unique())
# print(df['City'].unique())
# print(df['Country'].unique())
# print(df['Clicked on Ad'].unique())
# print(df['Male'].unique())

# --------------------------------------------Removing redundant data (if any) -----------------------
cols = ['Ad Topic Line', 'City', 'Country']
print('\n Unique Values: \n', df[cols].describe(include=['O']))
# We have too many unique values for all the 3 colums which will not allow a machine learning model to establish easily valuable relationships hence we will drop the columns
df_new = df.drop(['Ad Topic Line', 'City', 'Country'], axis=1)

# --------------------------------------------Check null values (no null values present)-----------------
# -------------------------------------------- output - 0  hence no duplicate/null records -----------
sns.heatmap(df.isnull(), yticklabels=False)
print('\n Null : ', df.isnull().sum())
print('\n Duplicate: ', df.duplicated().sum())

df_new = df_new.reindex(columns=[
    col for col in df_new.columns if col != 'Clicked on Ad'] + ['Clicked on Ad'])
df_new.columns = df_new.columns.str.replace(' ', '_')

print('\v New Head:\n', df_new.head())

df_new.to_csv('advertising_updated.csv')
