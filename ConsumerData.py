import pandas as pd

# ============================================================
# STEP 1: Load Consumer Data
# Update this path to wherever your CSV file is saved
# ============================================================
consumer = pd.read_csv('ConsumerData.csv')

print(f'Consumer: {consumer.shape[0]:,} rows x {consumer.shape[1]} columns')


# ============================================================
# STEP 2: Clean Consumer Data
# ============================================================
c = consumer.copy()

# --- Y/N flag columns -> encode to 1/0 ---
# NaN means 'not known', treated as 0
yn_cols_consumer = [
    'Charitable', 'Health', 'Political', 'Religious',
    'Gardening', 'HomeImprovement', 'HomeImprovementDIY',
    'CatOwner', 'DogOwner', 'OutdoorsGrouping',
    'SelfImprovement', 'MovieCollector', 'Photography',
    'AutoWork', 'Fishing', 'CampingHiking', 'HuntingShooting',
    'EnvironmentalIssues', 'InvestmentsForeign', 'BeautyCosmetics',
    'TVCable', 'WirelessCellularPhoneOwner', 'CreditCardUser'
]
for col in yn_cols_consumer:
    if col in c.columns:
        c[col] = c[col].map({'Y': 1, 'N': 0}).fillna(0).astype(int)

print('Y/N columns encoded.')

# --- Encode OwnerRenter ---
# O = Owner (1), R = Renter (0), NaN = unknown (0)
c['OwnerRenter'] = c['OwnerRenter'].map({'O': 1, 'R': 0}).fillna(0).astype(int)

# --- Encode MaritalStatus ---
# One-hot encoded so no false ordering is implied
marital_dummies = pd.get_dummies(c['MaritalStatus'], prefix='Marital', dummy_na=False)
c = pd.concat([c.drop(columns='MaritalStatus'), marital_dummies], axis=1)

print('OwnerRenter + MaritalStatus encoded.')

# --- Fill nulls in numeric columns ---
c['HouseholdSize']           = c['HouseholdSize'].fillna(c['HouseholdSize'].median())
c['NumberOfChildren']        = c['NumberOfChildren'].fillna(0)
c['NetWorth']                = c['NetWorth'].fillna(c['NetWorth'].median())
c['GrandChildren']           = c['GrandChildren'].fillna(0)
c['SingleParent']            = c['SingleParent'].fillna(0)
c['Veteran']                 = c['Veteran'].fillna(0)
c['VehicleKnownOwnedNumber'] = c['VehicleKnownOwnedNumber'].fillna(0)
c['MusicCollector']          = c['MusicCollector'].fillna(0)
c['EducationOnline']         = c['EducationOnline'].fillna(0)
# HomePurchaseDate left as-is (date column, decide later)

print('Numeric nulls filled.')

# --- Final check ---
remaining = c.isnull().sum()
remaining = remaining[remaining > 0]
print('Remaining nulls:', remaining.to_dict() if len(remaining) > 0 else 'None!')
print(f'\nCleaned consumer data: {c.shape[0]:,} rows x {c.shape[1]} columns')
print('\nAll columns:')
for col in c.columns:
    print(f'  {col}')


# ============================================================
# STEP 3: Save
# ============================================================
c.to_csv('consumer_clean.csv', index=False)
print('\nSaved: consumer_clean.csv')
print('\nNext step: once your friend has property_clean.csv ready,')
print('we can run the correlation analysis to find which consumer')
print('attributes matter most for predicting home type.')