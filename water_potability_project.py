# -*- coding: utf-8 -*-
"""
@author: Mialy Andrianarivony

"""

# libs 
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# For Scale
from sklearn.preprocessing import StandardScaler


# importing csv file and read 
data = pd.read_csv("water_quality1.csv")
print(type(data))

ph_data = data["Ph"]
turb_data = data["Turbidity"]
temp_data = data["Temperature"]
cond_data = data["Conductivity"]


# WHOLE Population Parameters
# Mean
print("Mean: ")
pop_mean = data.mean()
print(pop_mean)
print()

# Variance
print("Variance:")
pop_var = data.var()
print(pop_var)
print()

# Standard Deviaion
print("Standard Deviation:")
pop_std = data.std()
print(pop_std)
print()

# Mode
print("Mode:")
pop_mode = data.mode()
print(pop_std)
print()


# Mode
print("Median:")
pop_med = data.median()
print(pop_med)
print()


# ph column only
# Mean
pop_ph_mean = ph_data.mean()
print("Population pH Mean: ")
print(pop_ph_mean, "==> ", round(pop_ph_mean, 3))

# Variance
pop_ph_var = ph_data.var()
print("Population pH Variance: ")
print(pop_ph_var, "==> ", round(pop_ph_var, 3))

# Standard Deviation
pop_ph_std = ph_data.std()
print("Population pH Standard Deviation: ")
print(pop_ph_std, "==> ", round(pop_ph_std, 3))
print()

print("Mode of data:")
print("The most frequent pH value:", stats.mode(ph_data)[0])

#######################################################
group = data.groupby("Label")

print("Not Potable water: ")
print()
not_potable = group.get_group(0)

print("NOT Potable water mean: ")
print(not_potable.mean())
print()


print("Potable water: ")
potable = group.get_group(1)
print("Potable water mean: ")
print(potable.mean())
print()


##############################################

potable_turb = potable["Turbidity"]
not_potable_turb = not_potable["Turbidity"]

print("Histogram of each group: Potable and Not Potable")
sns.histplot(not_potable_turb, bins=5, color='grey', kde=False)
sns.histplot(potable_turb, bins=7, color='green', kde=False)


plt.title("Histogram of each group: Potable and Not Potable")
plt.xlabel("Turbidity")
plt.show()




##############################################################
#Graph 

print()
print("Visualize Distribution")

x_val = list(potable["Ph"])
print()
y_val = list(potable["Temperature"])
x_val2 = list(not_potable["Ph"])
y_val2 = list(not_potable["Temperature"])
print()

plt.plot(x_val, y_val,'g^')
plt.xlabel('pH Value', fontsize=15)
plt.xlim(0, 14)
plt.ylabel('Temperature', fontsize=15)
plt.ylim(0, 50)
plt.title('Relationship between potable water with temperature and pH value')
plt.show()

plt.plot(x_val2, y_val2,'r*')
plt.xlabel('pH Value', fontsize=15)
plt.ylabel('Temperature', fontsize=15)
plt.title('Relationship between non-potable water with temperature and pH value')
plt.show()
print()

###############################################################

# Take the first 500 values from each type of water 
N = 500
print("3d scatter plot")
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(projection='3d')

ax.scatter(potable["Ph"][:N], potable["Temperature"][:N], potable["Turbidity"][:N], marker='o')
ax.scatter(not_potable["Ph"][:N], not_potable["Temperature"][:N], not_potable["Turbidity"][:N], marker='^')

ax.set_xlabel('pH')
ax.set_ylabel('Temperature')
ax.set_zlabel('Turbidity')
plt.title("Relationship between ph, Temperature and Turbidity on water potability", size=18)
plt.show()
###############################################################

potable_pH = potable["Ph"]
not_potable_pH = not_potable["Ph"]

print("Standard (Gaussian) Normal Distribution for Potable water pH value") 

# Populate x domain 
potable_ph_mean = potable_pH.mean()
print("Potable pH Mean: ", potable_ph_mean)
potable_ph_std = potable_pH.std()
print("Potable pH Standard Deviation: ",potable_ph_std)

# Normal distribution
mu, sigma = potable_ph_mean, potable_ph_std # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

# Draw a plot for normal distribution
count, bins, ignored = plt.hist(s, 30, density=True)
plt.title("Normal Distribution for Potable water pH value")
plt.xlabel("pH Value")
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()

###############################################################
# MACHINE LEARNING
print("MACHINE LEARNING")

data = np.array(data)
X = data[:, :-1] # X: Independent features
print("X:", X)
y = data[:, -1] # y: Label column
 

print("Data Exploration via Heatmap")
# Column name list
column_names = ['Ph', 'Turbidity', 'Temperature', 'Conductivity']
Xdf = pd.DataFrame(X, columns=column_names)
ydf = pd.DataFrame(y)
Xydf = Xdf
Xydf['Potability'] = ydf 
print(ydf)

# Get the correlations of each feature in the dataset
correlation_matrix = Xydf.corr()


print('\nCorrelation matrix:')
print(correlation_matrix)
top_corr_features = correlation_matrix.index
plt.figure(figsize=(20,20))
# Generate plot heat map
g=sns.heatmap(Xydf[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

print()
##############################################################
print("Create SVM Classifier")

# Create SVM classifier
classifier = OneVsOneClassifier(LinearSVC(random_state=0, dual=False, max_iter=15000))

# You can create SVM classifier simply as follows.
classifier = LinearSVC(random_state=0, dual=False, max_iter=15000)



# Scale using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype(np.float64))
print("X_scaled: ")
print(X_scaled)

# Split the test data set for cross validation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)
print("X_train: ")
print(X_train)

# Train the classifier
classifier.fit(X_train, y_train)

# Predict output for X_test
y_test_pred = classifier.predict(X_test)


print("Cross-Validation Metrics ")
# Compute the F1 score of the SVM classifier
num_folds = 3
# Compute the F1 score of the SVM classifier
f1 = cross_val_score(classifier, X_test, y_test, scoring='f1_weighted', cv=num_folds)
print(f1)
print("\n(1) F1 score: " + str(round(100*f1.mean(), 2)) + "%")
accuracy_values = cross_val_score(classifier, X_test, y_test, scoring='accuracy', cv=num_folds)
print("\n(2) Accuracy:"+str(round(100*accuracy_values.mean(),2))+"%")
# Compute Precision
precision_values = cross_val_score(classifier, X_test, y_test, scoring='precision_weighted', cv=num_folds)
print("\n(3) Precision:"+str(round(100*precision_values.mean(),2))+"%")
# Compute Recall
recall_values = cross_val_score(classifier, X_test, y_test, scoring='recall_weighted', cv=num_folds)
print("\n(4) Recall:"+str(round(100*recall_values.mean(),2))+"%")
print()



##############################################################

# Predicition function
def predict_potability(input_data):
    print('\ninput_data for testing:', input_data)
    
    input_data = np.array(input_data)
    # Reshape input_data_encoded into 2D array
    input_data = input_data.reshape(1, -1) 
    
    # Standard scale the encoded input data
    input_data_scaled = scaler.transform(input_data.astype(np.float64))
    
    print('\ninput_data_encoded_scaled:\n', input_data_scaled)
    
    
    # Predict the output of the test data (1)
    predicted_class = classifier.predict(input_data_scaled)
    print('\npredicted_class:', predicted_class)
    
    if predicted_class == 0:
        return("Water is not potable")
    elif predicted_class == 1:
        return("Water is potable")


# Unknown data
unknown1 = [7.6, 446.3352942, 25.61048032, 251.3283198] # should yield 1
unknown2 = [0.77, 557.4868432,	15.30942078, 870.3411328] # 0
unknown3 = [9.36, 789.520797, 25.58681044, 700.6243535] # 0
# confusing data - ph around 7.5, temperature around recommendation
unknown4 = [7.350378987, 786.3922184, 10.58681044, 557.4956847] # 0 - yield 0
unknown5 = [7.36876853, 500.234242, 20.3434344, 203.902323] # 1 - yield 1
unknown6 = [7.186931122, 427.5645348, 17.89, 477.9949918] # 1 - yield 1
unknown7 = [7.086877326, 399.2848211, 19, 548.917718] # 0 - yield 1 (incorrect)

print(predict_potability(unknown1))
print(predict_potability(unknown2))
print(predict_potability(unknown3))
print(predict_potability(unknown4))
print(predict_potability(unknown5))
print(predict_potability(unknown6))
print(predict_potability(unknown7))
print()

