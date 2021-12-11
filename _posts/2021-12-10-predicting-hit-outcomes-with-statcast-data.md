# Predicting Hit Outcomes with Statcast Data

A few years ago I read an article by FiveThirtyEight, <a href = 'https://fivethirtyeight.com/features/the-new-science-of-hitting/'>The New Science of Hitting</a> which took advantage of then newly released Statcast data to demonstrate the importance of launch angle and exit velocity on outcomes at the plate. For those unfamiliar with the subject, launch angle "measures the vertical direction of the ball coming off the bat; a launch angle of zero degrees would be a flat line, with positive numbers indicating an upward ball flight and negative ones indicating a ball driven into the ground", while exit velocity measures the speed of the ball coming off the bat. In combination, these two metrics provide insight into the potential outcome of any batted ball. 

I decided that I wanted to take a look at whether I could harness Statcast data myself to accurately predict for any given ball in play, whether it would result in a hit or an out. As there is nothing new under the sun, a quick google search revealed that this <a href='https://tht.fangraphs.com/using-statcast-data-to-predict-hits/'> precise question was tackled by the folks at fangraphs back in April of 2016 </a> and it would be disingenuous to claim that I haven't read through their work and drawn some inspiration. That being said, the goal of this article is to walk through a more complete overview of the EDA, model selection, and evaluation involved in a machine learning project, with decisions arrived at based purely on data from the 2021 season. 

## Exploratory Data Analysis
```python
import pybaseball
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Setting the plot style in honor of the inspiring article
plt.style.use('fivethirtyeight')

#Import statcast data for 2021 season
df = pybaseball.statcast(start_dt='2021-01-01', end_dt='2021-11-02')
print(f'Total pitches in 2021: {df.shape[0]}')
```
Total pitches in 2021: 720637

The data provided by statcast contains 91 different fields containing information for each pitch on data points ranging from the pitch type, pitcher, and spin rate to the current score, who the fielders are, and whether anyone is on 3rd. Many of the fields are not particularly relevant to our analysis, so only the relevant fields are selected:

- Event (Double Play, Homerun, etc.)
- Hit Location Coordinates
- Estimated Batting Average
- Hit Distance
- Launch Angle
- Launch Speed (Exit Velocity)
- Fielding Alignments

```python
df = df[['events','hc_x', 'hc_y', 'estimated_ba_using_speedangle',
        'hit_distance_sc', 'launch_angle', 'launch_speed', 'if_fielding_alignment',
        'of_fielding_alignment']].copy()
```
Additionally, not all pitches result in a ball in play. Given we are predicting outcomes for balls put in play, let's drop any records where there is no batted ball.
```python
df.dropna(subset=['hc_x', 'hc_y','hit_distance_sc', 
                  'of_fielding_alignment','if_fielding_alignment'], how='any', inplace=True)
#Ensure data is numeric
df['launch_speed'] = df.launch_speed.astype(float)
df['launch_angle'] = df.launch_angle.astype(float)
print(f'Total pitches resulting in a ball in play: {df.shape[0]}')
```
Total pitches resulting in a ball in play: 122561
To begin, let's replicate the graph displayed in the FiveThirtyEight article - a scatterplot of hits by launch angle and speed, colored by the expected batting average. 
```python
df.rename({'estimated_ba_using_speedangle' : 'hit_probability'}, axis=1, inplace=True)
plt.figure(figsize=(7,9))
sns.scatterplot(x='launch_speed', y='launch_angle',
                hue=df.hit_probability.astype(float)*100, 
                palette='viridis', marker='|',
                data=df).set(title='Hit probability by Launch Angle and Speed')
```
<img src="/images/hit_prob_fivethirtyeight.png" alt="hi" class="inline"/>

It looks like not much has changed since 2016 - there is a pocket of high probability hits around 25 degree launch angle and >100mph exit velocity, with a streak of line drives stretching across the graph. 

What this doesn't show, however, is how the different outcomes fall within this space. In other words, what types of plays make up each of the "hot zones" in our 2x2. Let's break up the graph by event and see if a clear pattern starts to emerge.

```python
sns.relplot(
    data=df,
    x="launch_speed", y="launch_angle",
    hue=df.hit_probability.astype(float), col=df.events.astype(str),
    kind="scatter",
    palette='viridis', marker = '|',
    height=5, aspect=.75, facet_kws=dict(sharex=True),
    col_wrap=4
)
```
<img src="/images/hit_prob_relplot.png" alt="hi" class="inline"/>
Immediately we can see that some of the events group themselves around specific launch angle and speed combinations (e.g. Homeruns, Sac Flies, and Doubles), while others have a much broader distribution. For all groups however, there are fringes on the perimeter where the outcome differs from the expectation. Hits that occured in spite of Statcast's prediction, and outs where players were robbed of what would normally be a hit.

The data is pointing us towards what is intuitively obvious - there is more to whether or not a batted ball will be a hit than just the launch angle and speed. Let's take a look at how things look for the most obvious dimension - where a ball is hit. 

Statcast provides the x and y coordinates of each batted ball, which can be plotted to show the location of each hit. Boundaries vary slightly due to ballbark dimensions, but what we get is a beafutiful outline of the diamond colored by the hit probabilities at each location.
```python
fig, ax = plt.subplots(figsize = (16,10))
sns.scatterplot(x='hc_y', y='hc_x',
                hue=df.hit_probability.astype(float)*100, 
                palette='mako', marker='x',
                data=df, ax=ax)
```
<img src="/images/hit_prob_xy.png" alt="hi" class="inline"/>

There are a few areas with near certainties - hits over the outfield wall, foul balls, and infield hits/bunts. However, the majority of the graph displays a neapolitan type coloration, with blurred boundaries on hit probabilities. Let's add a third dimension (exit velocity) and see how the graph changes. 

```python
for azim in [30, 320]:
    fig = plt.figure(figsize = (16,20))
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(df.hc_x, df.hc_y, df.launch_speed, 
                 c=df.hit_probability.astype(float)*100,
                cmap='mako', marker='o')
    ax.view_init(5, azim)
```

<img src="/images/hit_prob_3dmako1.png" alt="hi" class="inline"/>

<img src="/images/hit_prob_3dmako2.png" alt="hi" class="inline"/>

We start to see some of the more convoluted areas in the original graph begin to seperate along the z-axis, exit velocity. How quickly a ball reaches its destination obviously has an impact on the outcome, with that relationship shown in our 3D rendering.

While we've uncovered a couple of the features that seem to influence the outcome of a hit, there's likely several other factors involved, creating an n-dimensional space that is more difficult to visualize. Once we've developed our model, we'll be able to uncover which features are most useful in predicting outcomes - but for now let's move on to preparing our dataset for model selection.
## Data Preparation
Before we split our data into our training and test sets, lets take a peek at how the different features are distributed.
```python
#Numerical Data Distributions
df.hist(bins=100, figsize=(10, 8))
plt.tight_layout()
```

<img src="/images/numerical_dist.png" alt="hi" class="inline"/>

```python
#Categorical Data Distributions
cat_data = df[['if_fielding_alignment', 'of_fielding_alignment']]

fig, axes = plt.subplots(2, figsize=(10,8), sharex=True)
for idx, col in enumerate(cat_data.columns):
    cat_data[col].value_counts(ascending=True, normalize=True).plot(kind='barh', 
                                                                    ax=axes[idx], title=col)
plt.tight_layout()
```
<img src="/images/cat_dist.png" alt="hi" class="inline"/>

Most of our numberical attributes have a fairly normal distribution, with the exception of Hit Distance. Infield fielding alignment is in a standard format only ~60% of the time, while the Outfield is in a standard format ~90% of the time.
Now, let's assign our target variable "outcome" and split our data, reserving 30% of the data as a holdout for testing. I explored using stratified splits to ensure accurate sampling of the infield shifts would be represented, however the size of the data set resulted in a similar output as random sampling - so we'll keep it simple.

```python
from sklearn.model_selection import train_test_split
#Encoding binary classification for hits and outs
df['outcome'] = np.where(df.events.isin(['field_out', 'double_play','force_out', 'sac_fly', 
                                            'grounded_into_double_play', 'fielders_choice_out',
                                            'fielders_choice', 'sac_bunt', 'triple_play',
                                            'sac_fly_double_play', 'sac_bunt_double_play']), 0, 1)
X_cols = ['hc_y', 'hc_x', 'hit_distance_sc', 'launch_speed', 'launch_angle',
        'if_fielding_alignment', 'of_fielding_alignment']
        
#Splitting Data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(df[X_cols], df.outcome,
                                                    test_size = 0.3, random_state = 42)
```
Next, we'll build a small pipeline for two important steps in data preparation: Categorical Encoding and Feature Scaling.

Categorical encoding is the process of transforming 'categorical' or text data to a numerical representation, while feature scaling normalizes numerical data to a comparable scale (e.g. 0-1). Both of these steps help to improve the models ability to accurately interpret the data down the road.
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import TransformerMixin
num_attributes = ['hc_x', 'hc_y', 'launch_angle', 'launch_speed', 'hit_distance_sc']
cat_attributes = ['if_fielding_alignment', 'of_fielding_alignment']

#Create custom transformer for updating of_fielding
class AttributeUpdater(TransformerMixin):
    def __init__(self, col_name='of_fielding_alignment'):
        self.update_dict = {'Standard' : 'Standard_of',
                            'Strategic' : 'Strategic_of'}
        self.col_name = col_name
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X[self.col_name] = X[self.col_name].apply(lambda x: self.update_dict.get(x, x))
        return X

#Categorical attribute pipeline
cat_pipe = Pipeline([
    ('of_update', AttributeUpdater()),
    ('encoder', OneHotEncoder())
])

#Pipeline for handling both categorical encoding and numerical scaling
full_pipeline = ColumnTransformer([
    ('numerical', StandardScaler(), num_attributes),
    ('categorical', cat_pipe, cat_attributes)
])

X_train_prep = full_pipeline.fit_transform(X_train)
```
## Model Selection
With our data prepared, we can move onto trying out different models - for the purpose of creating a baseline, we'll start with 4 different models:

- K Nearest Neighbors: Assigns new data points to a class based on the "k" nearest neighbors per the selected distance metric. In other words, a data point is assigned based on the classification of the other points closest to it. 

- Logistic Regression: Predicts the probability of a classification based on a logistic function, and can be used as a classifier based on cutoffs.

- Support Vector Machine: SVMs are powerful classification algorithms that essentially seperates points by a linear or non-linear boundary that creates the greatest margin between classes.

- Decision Tree: Decision Trees are one of the simplest models to understand, as they work by following a set of algorithmic decision points, similar to a flow diagram, with each path ending in a 'leaf' and an associated classification. 

Each model will be scored using cross validation, which is a method of sampling that splits the data into n-number of 'folds', training on n-1 of the folds and then evaluating on the remaining fold. The result is an array of scores, and a reduction in the likelihood of overfitting the model on the training set.

The scoring metric for evaluating our models will be what is called an F-score, using a harmonic mean to calculate the balance between precision and recall. In simpler terms:

- Precision is a measure of how 'precise' your model is. If you think of it as someone who is fishing in with a net in a lake with 100 fish, and they cast in a specific spot and pull back 20 fish and nothing else, they are very precise but not exactly comprehensive. 

- Recall is a measure of the coverage your model has. To rely on the fishing analogy, if you cast a much larger net across the whole lake and pull back all of the fish, but also a bunch of rocks and sticks, you will have a high recall and lower precision.

A perfect fisherman would be able to pull back 100 fish, but that is rarely the case in the real world. As such, the goal for a classification model is to minimize the number of errant hits and thus maximize your F-score.

We'll evaluate the F-Score and efficiency for each model before selecting a couple to fine-tune for our final model.
```python
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import time

from sklearn.model_selection import cross_val_score

logreg = LogisticRegression()
supv = svm.SVC()
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()

#Loop through candidate models, recording F-Score and time for cross validation
scores = []
for model_name, model in [('Regression', logreg),
                          ('SVM', supv),
                          ('KNN', knn),
                          ('Decision Tree', dtc)]:
    t1 = time.time()
    cross_val = cross_val_score(model, X_train_prep, y_train, scoring='f1')
    avg_fscore = cross_val.mean()*100.0
    t2 = time.time()
    time_elapsed = t2-t1
    print(f'Time Elapsed {model_name}: {time_elapsed:.2f}s')
    scores.append((model_name, avg_fscore, time_elapsed))

score_df = pd.DataFrame(scores, columns=['model', 'fscore', 'time_elapsed'])
score_df.sort_values(by=['fscore'], inplace=True)
fig, axes = plt.subplots(2, figsize=(10,10), sharey=True)
colors = ['tab:blue', 'tab:red']
for idx, measure in enumerate(['fscore', 'time_elapsed']):
    axes[idx].barh(score_df.model, score_df[measure], color=colors[idx])
    axes[idx].set_xlabel(f'{measure}')

plt.suptitle('Baseline Model Evaluation')
plt.tight_layout()
```
Time Elapsed Regression: 0.93s
Time Elapsed SVM: 641.62s
Time Elapsed KNN: 7.09s
Time Elapsed Decision Tree: 2.52s
<img src="/images/baseline_eval.png" alt="hi" class="inline"/>

The Logistic Regression performed the worst out of the candidates, with an score of only ~50%. Meanwhile, the KNN model outperformed the other models while also working very efficiently, which is a hallmark of the algorithm. SVM has a solid F-Score baseline, however it is far less efficient compared to the other models. Given the similar performance across KNN, SVM, and the Decision Tree, I'm going to go forward with optimizing each model to see if any has a significant performance improvement.

To do this, we'll do a Grid Search on each of the models to identify the optimal hyperparameters for the best F-Score.
```python
from sklearn.model_selection import GridSearchCV

param_grid_knn = [
    {'n_neighbors' : [5, 10, 50, 100, 150],
    'weights' : ['uniform', 'distance'],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}
]

param_grid_svm = [
    {'C': [1, 10, 100, 1000], 
    'gamma': [0.001, 0.01, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
]

param_grid_dtc = [
    {'criterion' : ['entropy', 'gini'],
    'max_depth': range(1,10),
    'min_samples_split': range(1,10),
    'min_samples_leaf' : range(1,5)}
]

#Grid Searching KNN and SVM for optimal estimator
best_models = {}
for model_name, model, params in [('KNN', knn, param_grid_knn),
                                 ('SVC', supv, param_grid_svm),
                                 ('DTC', dtc, param_grid_dtc)]:
    t1 = time.time()
    print(f'Grid Searching: {model_name}')
    grid = GridSearchCV(model, params, cv=2, scoring='f1', return_train_score=True, n_jobs=-1)
    grid.fit(X_train_prep, y_train)
    print(f'Grid Search Complete, Best Parameters: {grid.best_params_}')
    best_models[model_name] = grid.best_estimator_
    t2 = time.time()
    print(f'Total Time Elapsed: {(t2-t1)/60:.2f} min\n')
```
Grid Searching: KNN
Grid Search Complete, Best Parameters: {'algorithm': 'auto', 'n_neighbors': 10, 'weights': 'distance'}
Total Time Elapsed: 33.23 min

Grid Searching: SVC
Grid Search Complete, Best Parameters: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}
Total Time Elapsed: 6112.64 min

Grid Searching: DTC
Grid Search Complete, Best Parameters: {'criterion': 'gini', 'max_depth': 9, 'min_samples_leaf': 4, 'min_samples_split': 9}
Total Time Elapsed: 1.00 min

After some lenghty run times due to the Support Vector Classifier, we're presented with the optimal combination of hyperparameters based on the grids provided. Now we'll see how the best models perform on the training set.

<img src="/images/update_eval.png" alt="hi" class="inline"/>

A slight improvement for all of the models, with the SVC just barely edging out KNN as having the best score. Due to the lengthy run-time for evaluating this model in comparison to the others, it doesn't appear to be the optimal solution. Nonetheless, now that we have the model built we'll see how it performs on the test set just as well.
## Evaluating the Test Set
With our models selected, it's time to see how they perform on the test set. We'll apply the pipeline to our hold-out data, and then score each model accordingly and see how it generalizes to data it's never seen.
```python
from sklearn.metrics import f1_score
X_test_prep = full_pipeline.transform(X_test)

for model_name, model in best_models.items():
    t1 = time.time()
    y_pred = model.predict(X_test_prep)
    f_score = f1_score(y_test, y_pred)
    t2 = time.time()
    print(f'{model_name} F-Score: {f_score:.2%}')
```
KNN F-Score: 84.45%
SVC F-Score: 85.85%
DTC F-Score: 82.61%

The scores are in line with the scores we expected based on our training set, with the Support Vector Classifier and K-Nearest Neighbors performing the best at an F-Score of ~85%. Out of curiousity, we'll take a look at where the ~15% wrongly classified instances are occuring. 

For this data:

- A False Positive is an instance that resulted in an out, but our model predicted would be a hit
- A False Negative is an instance that resulted in a hit, but our model predicted would be an out

```python
for model_name, model in best_models.items():
    y_pred = model.predict(X_test_prep)
    graphing = X_test.copy()
    graphing['y_test'] = y_test
    graphing['y_pred'] = y_pred
    graphing['fp'] = np.where((graphing.y_test == 0) & (graphing.y_pred == 1), 1, 0)
    graphing['fn'] = np.where((graphing.y_test == 1) & (graphing.y_pred == 0), 0.5, 0)
    graphing['diff_'] = graphing.fp + graphing.fn

    fig, ax = plt.subplots(figsize = (16,10))
    sns.scatterplot(x='hc_y', y='hc_x',
                    hue=graphing.diff_, 
                    palette='hot_r',
                    data=graphing, ax=ax)

    ax.legend(ax.get_legend_handles_labels()[0], ['Correctly Classified',
                                                  'False Negative', 'False Positive'])
    ax.set_title(f'{model_name}')
```

<img src="/images/knn_heat.png" alt="hi" class="inline"/>

<img src="/images/svc_heat.png" alt="hi" class="inline"/>

<img src="/images/dtc_heat.png" alt="hi" class="inline"/>

All of the models seem to struggle with certain infield hits that the model expects to be an out, but result in hits. This could likely be attributed to seeing-eye singles, runners beating out the throw, or other instances of that sort. All of the models also seem to struggle with deep hits in the outfield - though the makeup is slightly different.

Both KNN and the SVC have a larger number of False Positives, while the Decision Tree seems to skew towards False Negatives. It appears something in the rule set is biasing the Decision Tree against predicting those fringe drives as hits, while the other two models are more likely to falsely predict a hit.
## Conclusion
We were able to build a couple of models which are able to accurately predict hit outcomes at ~85% based on statcast data, which is pretty good performance. If I was going to deploy a model out of the ones we trained, I would lean towards the KNN for it's accuracy and quick performance. While the Support Vector Classifier performed slightly better, the lengthy run-time would make it less than optimal for a scenario where you are estimating outcomes (e.g. during a live broadcast).

The models struggled on classifying hits in the deep outfield and the infield, and if I were to revist the subject I would likely select a few other features and see how they impact the performance, such as ball-park and player at-bat. In another iteration, it would also be interesting to build a model that provides a probability statistic (liklihood of a hit) or predicts the actual event (single, HR, out, etc.), but for a baseline approach we've developed a solid model.



