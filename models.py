import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def data_clean(df_attribute, df_category, df_business, bus_type):
	""" Cleans up the attribute and category data. Returns the data for modeling as a pivot table. """

	# Unpack attributes
	df_dirty = df_attribute[['name', 'value']].reset_index()

	# Attach code columns
	df_dirty['name_code'] = df_dirty['name'].astype('category').cat.codes
	df_dirty['value_code'] = -1

	# Convert all of the values to ints
	for i, row in df_dirty[['name_code']].drop_duplicates().iterrows():
	    n = row['name_code']
	    df_dirty.loc[df_dirty['name_code']==n, 'value_code'] = \
		df_dirty.loc[df_dirty['name_code'] == n]['value'].astype('category').cat.codes

	df_pivot = pd.pivot_table(df_dirty, index='business_id', columns='name_code', values='value_code', fill_value=-1)

	# Attach the categories
	df = df_category[['category']].loc[df_category.loc[df_category['category'] == bus_type].index].reset_index()
	df_ok_cats = df.groupby(['category'])[['business_id']].count().rename({'business_id': 'total'}, axis=1)
	df_ok_cats = df_ok_cats.loc[df_ok_cats['total'] > 500].sort_values('total', ascending=False)
	df = pd.merge(df, df_ok_cats, left_on=['category'], right_index=True)

	df.drop(columns=['total'], inplace=True)
	df.rename(columns={'category': 'value'}, inplace=True)

	df['there'] = 1
	df = pd.pivot_table(df, index='business_id', columns='value', values='there', fill_value=0)
	df_pivot = pd.merge(df, df_pivot, left_index=True, right_index=True)

	# Attach the star ratings
	df = pd.merge(df_category.loc[df_category['category'] == bus_type], df_business[['stars', 'review_count']],
		      left_index=True, right_index=True)
	df_clean = pd.merge(df_pivot, df[['stars', 'review_count']], left_index=True, right_index=True)

	return df_clean


def fill_missing_values(df_clean, good_attr, missing_val=-1):
	""" Modifies the contents of df_clean by calculating the mode of each categorical value and filing in the missing data. """

	for c in good_attr:
		m = df_clean[c].loc[df_clean[c] != missing_val].mode()[0]
		df_clean[c] = df_clean[c].apply(lambda x: {-1: m}.get(x, x))

def identify_good_features(df_clean, threshold=0.5):
	""" Goes through the dataframe that was created for modeling, and identifies which features are useful. """

	good_attr = []
	for idx in df_clean.columns.values:
		if len(df_clean.loc[df_clean[idx] == -1]) / len(df_clean) < threshold:
			good_attr.append(idx)
	good_attr.remove('stars')
	good_attr.remove('review_count')
	return good_attr


def train_rf_model(df_clean, good_attr, target_col='stars', weight_col='review_count', folds=5, n_iter=20, n_jobs=1):
	""" Uses a random grid search with K-fold cross validation to build a good random forest model. """

	# Set the ranges for the hyperparameter fitting
	n_estimators = list(range(100, 500, 20))
	max_features = ['auto', 'sqrt']
	max_depth = list(range(4, 10, 1))
	max_depth.append(None)
	min_samples_split = [2, 5, 10]
	min_samples_leaf = [1, 2, 4]

	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
		       'max_features': max_features,
		       'max_depth': max_depth,
		       'min_samples_split': min_samples_split,
		       'min_samples_leaf': min_samples_leaf}

	pprint(random_grid)

	# Use the random grid to search for best hyperparameters
	rf = RandomForestRegressor()
	rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=n_iter, cv=folds, verbose=2, random_state=0, n_jobs=n_jobs)

	# Fit the random search model
	rf_random.fit(df_clean[good_attr], df_clean[target_col], sample_weight=df_clean[weight_col].astype(int))

	return rf_random


def train_naive_bayes(df_clean, good_attr, target_col='stars', weight_col='review_count', folds=5, n_iter=20, n_jobs=1):
	""" Uses a random grid search with K-fold cross validation to build and test a naive bayes model. Note that
	this model assumes the target values are integers. """

	nb = GaussianNB()
	nb_random = GridSearchCV(estimator=nb, param_grid={}, cv=folds, verbose=2, n_jobs=n_jobs)

	# Fit the random search model
	nb_random.fit(df_clean[good_attr], (df_clean[target_col]*10).astype(int), sample_weight=df_clean[weight_col].astype(int))

	return nb_random


def apply_model(df_clean, model, output_col='preds', target_col='stars', weight_col='review_count'):
	""" Calculates the predictions on df_clean of model. The results are stored back into df_clean under output_col.
	Returns the weighted KPI. """

	# Try a weighted kpi
	df_clean[output_col] = model.predict(df_clean[good_attr])
	df_clean[output_col] = (df_clean[output_col]*2).round(decimals=0)*0.5
	return sum((df_clean[target_col] - df_clean[output_col])**2*df_clean[weight_col])**0.5 / sum(df_clean[weight_col])

def rf_feature_importance(df_clean, model):

	res = list(zip(df_clean.columns.values, model.feature_importances_))
	res.sort(key=lambda x: x[1])
	return res


# Read in the data

# Pull data some raw data
print('Reading data files...')
df_business = pd.read_csv('data/business.csv')
#df_review = pd.read_csv('data/review.csv')
df_attribute = pd.read_csv('data/attribute.csv')
df_category = pd.read_csv('data/category.csv')

# Add some indices
#df_review.set_index(['business_id'], inplace=True)
df_business.set_index(['id'], inplace=True)
df_category.set_index(['business_id'], inplace=True)
df_attribute.set_index(['business_id'], inplace=True)

print('Cleaning data...')
df_clean = data_clean(df_attribute, df_category, df_business, bus_type='Restaurants')
good_attr = identify_good_features(df_clean, threshold=0.5)
#fill_missing_values(df_clean, good_attr, missing_val=-1)  # This is removed, as it seems to decrease model accuracy!

print('Creating RF model...')
rf_random = train_rf_model(df_clean, good_attr, target_col='stars', weight_col='review_count', folds=5, n_iter=20, n_jobs=4)
kpi = apply_model(df_clean, rf_random, output_col='preds', target_col='stars', weight_col='review_count')
print('Got a KPI value of {} for the random forest.'.format(kpi))

print('Creating NB model...')
nb_random = train_naive_bayes(df_clean, good_attr, target_col='stars', weight_col='review_count', folds=5, n_iter=20, n_jobs=4)
kpi = apply_model(df_clean, nb_random, output_col='preds', target_col='stars', weight_col='review_count')
print('Got a KPI value of {} for the naive bayes model.'.format(kpi))

print('The feature importances for the rf model are:')
res = rf_feature_importance(df_clean, rf_random.best_estimator_)
pprint(res)

