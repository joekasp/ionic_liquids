# Main Idea
The main idea of this project is to design a machine learning framework to take some easily-measured properties of the ionic liquid to predict some non-trivial properties.


# Work flow

#### 1. Date collection and cleaning
Collect data from ILthermo, catenate different data into one csv file which is ready for machine learning.
#### 2. Descriptors generation
1. Find SMILES string of all the cations and anions in the binary system.
2. Feed the SMILES string into RDkit and generate molecular discriptors for machine learning.
3. The final data set should be a N * M data sheet, where N is number of data points, and M is the dimension of features or number of discriptors.
4. Data can be split into testing, training, validation data.

#### 3. Machine Learning
Use neural network from scikit learn or tensor flow 

#### 4. Visualization

# Use Cases
This package aims to help researchers choose potential
binary systems of ionic liquids by predicting desired properties
of the resulting mixture.

## Electrical Conductivity Prediction
The electrical conductivity is a non-trivial property for measurement. However, it is of the great importance in the application of flow battery.
In this project, we will use some simple properties, e.g. density, dielectric constant, ratio of the binary component to output the electrical conductivity.

