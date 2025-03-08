## Dataset
The dataset is a compilation of water quality metrics for over 8500 different water bodies. No values are missing and data is clean.

Columns:
1. pH value:
PH is an important parameter in evaluating the acid–base balance of water. It is also
the indicator of acidic or alkaline condition of water status. WHO has recommended
maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges
were 6.52–6.83 which are in the range of WHO standards.
2. Turbidity:
The turbidity of water depends on the quantity of solid matter present in the
suspended state. It is a measure of light emitting properties of water and the test is
used to indicate the quality of waste discharge with respect to colloidal matter.
3. Temperature:
For public water supply, it should be between 10 to 15.6 degree celcius. The
temperature greater than 25 degree is undesirable and above 30 degree is unfit for
public water supply.
4. Conductivity:
Pure water is not a good conductor of electric current rather a good insulator. Increase
in ions concentration enhances the electrical conductivity of water. Generally, the
amount of dissolved solids in water determines the electrical conductivity. Electrical
conductivity (EC) actually measures the ionic process of a solution that enables it to
transmit current. According to WHO standards, EC value should not exceed 400
μS/cm.
5. Label:
Indicates if water is safe for human consumption where 1 means Potable and 0 means
Not potable.
This gives an insight to the water supplies available in a region and estimate the amount of
water that are safe for human consumption. It is important to know the potability as it can
lead to diseases and other harmful effects on the human population if they are not constantly
monitored and measured.

With this dataset as well, we can perform data prediction of whether the water is potable or to fill in missing values of other records.
Load the CSV file onto python using the pandas library, pd.read_csv method so that the data is loaded as Dataframe to make calculation easier later.

This dataset is available in [Kaggle](https://www.kaggle.com/datasets/vishnuverma1441/real-time-water-quality)

# Data Analytics
## Population
To start the Data Analysis, perform calculations on the entire population to see the Mean,
Median, Mode, Variance and Standard Deviation. This can then be to compare with the
potable and non-potable water.

## Grouping
Then group the data into 2: potable and non-potable water. Grouping can be easily create with the groupby() method by pandas.
These can stored in different variables to easily calculate the statistics for one specific group only.
Variables: potable and not_potable now store the two different groups.

## Data Visualization
To find if there are at all any correlation between potability and other features, these graphs may help achieve this:
• Histogram of potable and non-potable water and the Turbidity, to see the count and
distribution of data

• Scatter plot – of pH and temperature
There is a clear range set for the pH and the Temperature of potable water. Ph value is
set in the range 6.5 to 8.5 while the temperature is between 10 to 30 Celcius.
However, looking at the non-potable water, the pH Value and the Temperature do no
really matter and therefore these features will not be able to predict the potability
alone.

• 3D Scatter Plot
When the 3 features are plotted together, there is a clear section for acceptable potable water.

• Normal Distribution – Potable water’s pH values. This will help see how potable
water’s pH values are distributed around the mean

## Machine Learning
1. Heat Map – see the correlation between the features and the label (potability). It will
be possible to see if there is any sort of linear relationship for prediction or a way to
easily know if a water body is potable or not.
Using the pandas method corr() to return the correlation matrix of the entire data.

2. Classifier – Using sk.learn to create a Linear Regression Classifier.
Then Scale the entire dataset to evenly weight all features and minimize outliers.
Split the data for training and testing. 20% for testing should be good for this dataset.
Calculate the Cross-Validation Metrics (F1 score, Accuracy, Precision and Recall
values)
This will help see if the model has successfully made sense of the data and can be
used for predicting other data.

3. Prediction – Use scaler and classifier to predict the potability of other datasets
