import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selectionimport train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_modelimport LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selectionimport GridSearchCV
df = pd.read_csv('../input/adult.csv')
df.head()
df.shape
df.info()
df.duplicated().sum()
df = df.drop_duplicates()
df.shape
cat_feat = df.select_dtypes(include=['object']).columns
cat_feat
print('% of missing values :')
for c in cat_feat:
perc = len(df[df[c] == '?']) / df.shape[0] * 100
print(c, f'{perc:.1f} %')
df.describe()
###Taking a look at the target (income) without distinction of sex
print(f"Ratio above 50k : {(df['income'] == '>50K').astype('int').sum() / df.shape[0] * 100 :.2f}%")
num_feat = df.select_dtypes(include=['int64']).columns
num_feat
###Plot pairwise relationships in a dataset
plt.figure(1, figsize=(16,10))
sns.pairplot(data=df, hue='sex')
plt.show()
plt.figure(figsize=(18,10))
plt.subplot(231)
i=0
for c in num_feat:
plt.subplot(2, 3, i+1)
i += 1
sns.kdeplot(df[df['sex'] == 'Female'][c], shade=True, )
sns.kdeplot(df[df['sex'] == 'Male'][c], shade=False)
plt.title(c)
plt.show()
plt.figure(figsize=(18,25))
plt.subplot(521)
i=0
for c in cat_feat:
plt.subplot(5, 2, i+1)
i += 1
sns.countplot(x=c, data=df, hue='sex')
plt.title(c)
plt.show()
###nb of female / male
nb_female = (df.sex == 'Female').astype('int').sum()
nb_male = (df.sex == 'Male').astype('int').sum()
nb_female, nb_male
###nb of people earning more or less than 50k per gender
nb_male_above = len(df[(df.income == '>50K') & (df.sex == 'Male')])
nb_male_below = len(df[(df.income == '<=50K') & (df.sex == 'Male')])
nb_female_above = len(df[(df.income == '>50K') & (df.sex == 'Female')])
nb_female_below = len(df[(df.income == '<=50K') & (df.sex == 'Female')])
nb_male_above, nb_male_below, nb_female_above, nb_female_below
print(f'Among Males : {nb_male_above/nb_male*100:.0f}% earn >50K // {nb_male_below/nb_male*100:.0f}% earn <=50K')
print(f'Among Females : {nb_female_above/nb_female*100:.0f}% earn >50K // {nb_female_below/nb_female*100:.0f}% earn <=50K')
###normalization
nb_male_above /= nb_male
nb_male_below /= nb_male
nb_female_above /= nb_female
nb_female_below /= nb_female
nb_male_above, nb_male_below, nb_female_above, nb_female_below
print(f'Among people earning >50K : {nb_male_above / (nb_male_above + nb_female_above) *100 :.0f}% are Females and {nb_female_above / (nb_female*100:.0f}% earn <=50K')
print(f'Among people earning =<50K : {nb_male_below / (nb_male_below + nb_female_below) *100 :.0f}% are Females and {nb_female_below / (nb_female*100:.0f}% earn <=50K')
df['US native'] = (df['native.country'] == 'United‐States').astype('int')
plt.figure(figsize=(6,4))
sns.countplot(x='US native', data=df, hue='sex')
plt.show()
plt. figure(figsize=(6,4))
sns. countplot(x='income', data=df, hue='US native')
plt.show()
###nb of people earning more or less than 50k per origin
hb_native_above
- len(df[(df.income ==
'>50K') & (df['US native'] == 1)])
nb_native_below = lendf[(df.income ==
'<=50K') & (df['US native'] == 1)])
nb_ foreign_above = len(dfE (df.income ==
nb_foreign_below = len(df[(df.income ==
'<=50K') & (df['US native')
no_native_above, nb_native_below, nb_foreign_above, nb_foreign_below
nb_native = (df['US native'] == 1).astype( 'int').sum()
nb_foreign = df.shape[0] - nb_native
nb_native, nb_foreign
print(f'Among natives:(nb_native_above/nb_native*100:.0f)% earn >50K // (nb_native_below/nb_native*100:.0f)% earn <=50K’)
print (f'Among foreigners : (nb_foreign_above/nb_foreign*100:.0f)% earn ›50K // (nb_foreign_below/nb_foreign*100:.0f)% earn <=50K’)
###normalization
no_native_above / nb_native nb_native_below / =nb_native
nb_foreign_above /= nb_foreign nb foreign_below /= nb_foreign nb_native_above, nb_native_below, nb_foreign_above, nb_foreign_below
