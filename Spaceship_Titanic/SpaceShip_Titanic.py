#!/usr/bin/env python
# coding: utf-8

# # Spaceship Titanic with TFDF( TensorFlow Digital Forest API)

# ##

# ##### This notebook walks you through how to train a baseline Random Forest model using TensorFlow Decision Forests on the Spaceship Titanic dataset made available for this competition.

# #

# In[106]:


import tensorflow_decision_forests as tfdf
import pandas as pd

dataset = pd.read_csv("sample_submission.csv")
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="Transported")

model = tfdf.keras.RandomForestModel()
model.fit(tf_dataset)

print(model.summary())


# #

# ## Import the Library

# In[41]:


import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[42]:


print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)


# #

# ## Load the Data 

# In[43]:


dataset_df = pd.read_csv("train.csv")
print("Full train dataset shape is {}".format(dataset_df.shape))


# #

# ##### The data is composed of 14 columns and 8693 entries. Now let's try printing out the first 5 entries.

# In[44]:


dataset_df.head()


# #

# ##### There are 12 feature columns. Using these features the model has to predict whether the passenger is rescued or not indicated by the column ```Transported```.

# #

# ## Let do basic exploration of the dataset

# In[45]:


dataset_df.describe()


# In[57]:


dataset_df.info()


# ## Bar chart for label column: Transported

# In[47]:


plot_df = dataset_df.Transported.value_counts()
plot_df.plot(kind="bar")


# ## Normal data Distribution 

# In[48]:


fig, ax = plt.subplots(5,1,  figsize=(10, 10))
plt.subplots_adjust(top = 2)

sns.histplot(dataset_df['Age'], color='b', bins=50, ax=ax[0]);
sns.histplot(dataset_df['FoodCourt'], color='b', bins=50, ax=ax[1]);
sns.histplot(dataset_df['ShoppingMall'], color='b', bins=50, ax=ax[2]);
sns.histplot(dataset_df['Spa'], color='b', bins=50, ax=ax[3]);
sns.histplot(dataset_df['VRDeck'], color='b', bins=50, ax=ax[4]);


# ## Prepare the DataSet 

# #

# ### Drop both PassengerId and Name columns as they are not necessary for model training.

# In[65]:


print(dataset_df.columns)
dataset_df.head(5)


# ### Check for the missing values Cols

# In[85]:


cols_with_missing = [col for col in dataset_df.columns if dataset_df[col].isnull().any()]
cols_with_missing = sorted(cols_with_missing, reverse=True)  # Sort the list in descending order
print(cols_with_missing)


# ### Numerical Visualizing 

# In[84]:


dataset_df.isnull().sum().sort_values(ascending=False)


# #

# ##### This dataset contains a mix of numeric, categorical and missing features. TF-DF supports all these feature types natively, and no preprocessing is required. 
# 
# ##### But this datatset also has boolean fields with missing values. TF-DF doesn't support boolean fields yet. So we need to convert those fields into int. To account for the missing values in the boolean fields, we will replace them with zero. In this notebook, we will replace null value entries with zero for numerical columns as well and only let TF-DF handle the missing values in categorical columns. 
# 
# ##### Note: You can choose to let TF-DF handle missing values in numerical columns if need be.

# In[89]:


dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
dataset_df.isnull().sum().sort_values(ascending=False )


# #

# ##### Since, TF-DF cannot handle boolean columns, we will have to adjust the labels in column ```Transported``` to convert them into the integer format that TF-DF expects.

# In[90]:


label = "Transported"
dataset_df[label] = dataset_df[label].astype(int)


# #

# ##### We will also convert the boolean fields CryoSleep and VIP to int.

# In[91]:


dataset_df['VIP'] = dataset_df['VIP'].astype(int)
dataset_df['CryoSleep'] = dataset_df['CryoSleep'].astype(int)


# #

# ##### The value of column Cabin is a string with the format Deck/Cabin_num/Side. Here we will split the Cabin column and create 3 new columns Deck, Cabin_num and Side, since it will be easier to train the model on those individual data.
# 

# In[92]:


dataset_df[["Deck", "Cabin_num", "Side"]] = dataset_df["Cabin"].str.split("/", expand=True)


# #

# ##### Remove original Cabin column from the dataset since it's not needed anymore.

# In[93]:


try:
    dataset_df = dataset_df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")


# In[97]:


dataset_df.head(5)


# #

# ##### Now let us split the dataset into training and testing datasets:

# In[100]:


def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))


# #

# ##### There's one more step required before we can train the model. We need to convert the datatset from Pandas format (pd.DataFrame) into TensorFlow Datasets format (tf.data.Dataset).
# 
# ##### TensorFlow Datasets is a high performance data loading library which is helpful when training neural networks with accelerators like GPUs and TPUs.

# In[111]:


train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label)


# #

# ##### Select a Model There are several tree-based models for you to choose from.
# ##### RandomForestModeL, GradientBoostedTreesModel, GartModel, DistributedGradientBoostedTreesModel
# ##### To start, we'll work with a Random Forest. This is the most well-known of the Decision Forest training algorithms. 
# 
# ##### A Random Forest is a collection of decision trees, each trained independently on a random subset of the training dataset (sampled with replacement). The algorithm is unique in that it is robust to overfitting, and easy to use

# In[112]:


tfdf.keras.get_all_models()


# #

# ### Configure the Model

# In[115]:


rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")


# #

# ### Create a Random Forest

# In[116]:


rf = tfdf.keras.RandomForestModel()
rf.compile(metrics=["accuracy"]) # Optional, you can use this to include a list of eval metrics


# #

# ### Train the model
# ##### We will train the model using a one-liner.

# In[117]:


rf.fit(x=train_ds)


# #

# ### Visualize the model
# ##### One benefit of tree-based models is that we can easily visualize them. The default number of trees used in the Random Forests is 300. We can select a tree to display below.

# In[119]:


tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)


# #

# ### Evaluate the model on the Out of bag (OOB) data and the validation dataset
# ##### Before training the dataset we have manually seperated 20% of the dataset for validation named as valid_ds.
# 
# ##### We can also use Out of bag (OOB) score to validate our RandomForestModel. To train a Random Forest Model, a set of random samples from training set are choosen by the algorithm and the rest of the samples are used to finetune the model.The subset of data that is not chosen is known as Out of bag data (OOB). OOB score is computed on the OOB data.

# In[120]:


import matplotlib.pyplot as plt
logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.show()


# #

# ##### see some general stats on the OOB dataset

# In[121]:


inspector = rf.make_inspector()
inspector.evaluation()


# In[122]:


evaluation = rf.evaluate(x=valid_ds,return_dict=True)

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")


# #

# ### Variable importances
# ##### Variable importances generally indicate how much a feature contributes to the model predictions or quality. There are several ways to identify important features using TensorFlow Decision Forests. Let us list the available Variable Importances for Decision Trees:

# In[123]:


print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)


# #

# ##### As an example, let us display the important features for the Variable Importance NUM_AS_ROOT.
# ##### The larger the importance score for NUM_AS_ROOT, the more impact it has on the outcome of the model.
# ##### By default, the list is sorted from the most important to the least. From the output you can infer that the feature at the top of the list is used as the root node in most number of trees in the random forest than any other feature.

# In[124]:


# Each line is: (feature name, (index of the feature), importance score)
inspector.variable_importances()["NUM_AS_ROOT"]


# #

# ### Submission

# In[125]:


# Load the test dataset
test_df = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
submission_id = test_df.PassengerId

# Replace NaN values with zero
test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)

# Creating New Features - Deck, Cabin_num and Side from the column Cabin and remove Cabin
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
test_df = test_df.drop('Cabin', axis=1)

# Convert boolean to 1's and 0's
test_df['VIP'] = test_df['VIP'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

# Convert pd dataframe to tf dataset
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

# Get the predictions for testdata
predictions = rf.predict(test_ds)
n_predictions = (predictions > 0.5).astype(bool)
output = pd.DataFrame({'PassengerId': submission_id,
                       'Transported': n_predictions.squeeze()})

output.head()


# In[126]:


sample_submission_df = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')
sample_submission_df['Transported'] = n_predictions
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission_df.head()


# In[ ]:




