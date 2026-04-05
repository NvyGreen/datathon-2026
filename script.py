import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, NearestNeighbors
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# Consumer Data (cleaned)
  # HouseholdSize
  # VehicleKnownOwnedNumber
  # NetWorth
  # Gardening
  # HomeImprovement
  # OutdoorsGrouping
  # OwnerRenter

# Property Assessment Data (cleaned)
  # NumberOfBedrooms
  # GarageParkingNumberOfCars
  # TotalAssessedValue
  # LotSizeOrArea
  # YearBuilt
  # SalesPriceFromAssessment

consumer_data_columns = ['HouseholdSize', 'VehicleKnownOwnedNumber', 'NetWorth', 'Gardening', 
                         'HomeImprovement', 'OutdoorsGrouping', 'OwnerRenter']

property_data_columns = ['NumberOfBedrooms', 'GarageParkingNumberOfCars', 'TotalAssessedValue', 
                         'LotSizeOrArea', 'YearBuilt', 'LotSizeOrArea', 'SalesPriceFromAssessment']

consumer_data = pd.read_csv('consumer_clean.csv')
property_data = pd.read_csv('PropertyAssessmentData_cleaned.csv')

consumer_data['Address'] = consumer_data['Address'].str.strip().str.lower()
property_data['PropertyAddress'] = property_data['PropertyAddress'].str.strip().str.lower()

merged_data = pd.merge(consumer_data, property_data, left_on='Address', right_on='PropertyAddress', how='inner')

X = merged_data[consumer_data]
y = merged_data[property_data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

property_scaled = scaler.fit_transform(X_train)
consumer_scaled = scaler.fit_transform(X_test)

nn = NearestNeighbors(n_neighbors=3, metric='cosine')
nn.fit(property_scaled)

distances, indices = nn.kneighbors(consumer_scaled)

# top_k_houses = property.iloc[indices[0]].copy()
# top_k_houses["Similarity"] = 1 - distances[0]
# print(top_k_houses)



knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy_percentage = knn.score(X_test, y_test)
print(accuracy_percentage)

cm = confusion_matrix(y_test, y_pred)
print(cm)

cr = classification_report(y_test, y_pred)
print(cr)
