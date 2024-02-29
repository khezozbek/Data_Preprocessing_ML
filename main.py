# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Wine Quality Red dataset
dataset = pd.read_csv('winequality-red.csv', delimiter=';')

# Separate features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Split the dataset into an 80-20 training-test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



# Create an instance of the StandardScaler class
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# Fit the StandardScaler on the features from the training set and transform it
X_train = sc.fit_transform(X_train)

# Apply the transform to the test set
X_test = sc.transform(X_test)

# Print the scaled training and test datasets

print(X_train)
print(X_test)