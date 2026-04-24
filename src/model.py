import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# ============================================================
# STEP 1: Load Cleaned Data
# ============================================================
consumer = pd.read_csv('consumer_clean.csv')
prop     = pd.read_csv('PropertyAssessmentData_cleaned.csv')

print(f'Consumer: {consumer.shape[0]:,} rows x {consumer.shape[1]} columns')
print(f'Property: {prop.shape[0]:,} rows x {prop.shape[1]} columns')

# ============================================================
# STEP 2: Prepare Consumer Features
# Drop Address — identifier only, not a feature
# Convert boolean Marital columns to int
# ============================================================
consumer_clean = consumer.copy()
if 'Address' in consumer_clean.columns:
    consumer_clean = consumer_clean.drop(columns=['Address'])

for col in ['Marital_S', 'Marital_M']:
    if col in consumer_clean.columns:
        consumer_clean[col] = consumer_clean[col].astype(int)

consumer_feature_cols = [col for col in consumer_clean.columns if col != 'MAK']
print(f'\nConsumer features ({len(consumer_feature_cols)}): {consumer_feature_cols}')

# ============================================================
# STEP 3: Prepare Property Features
# ============================================================
p = prop.copy()
p['HasPool']   = (p['PoolType'] != 'No Pool').astype(int)
p['HasGarage'] = (p['GarageTypeParking'] != 'No Garage').astype(int)
p['NumberOfBaths'] = p['NumberOfBaths'].fillna(p['NumberOfBaths'].median())
p['LotSizeOrArea'] = p['LotSizeOrArea'].fillna(p['LotSizeOrArea'].median())

property_feature_cols = [
    'NumberOfBedrooms', 'NumberOfBaths', 'LotSizeOrArea',
    'TotalAssessedValue', 'SalesPriceFromAssessment',
    'YearBuilt', 'GarageParkingNumberOfCars', 'HasPool'
]
property_feature_cols = [c for c in property_feature_cols if c in p.columns]
print(f'Property features ({len(property_feature_cols)}): {property_feature_cols}')

# ============================================================
# STEP 4: Merge Known Pairs for Train/Test
# 1,120 consumers whose MAK exists in both datasets
# ============================================================
matched = pd.merge(consumer_clean, p[['MAK'] + property_feature_cols],
                   on='MAK', how='inner')
print(f'\nMatched consumer-property pairs: {len(matched):,}')

# ============================================================
# STEP 5: 80/20 Train/Test Split
# Test set = consumers with a known property we can verify
# ============================================================
train_df, test_df = train_test_split(matched, test_size=0.2, random_state=42)
print(f'Training set: {len(train_df):,} (80%)')
print(f'Testing set:  {len(test_df):,} (20%)')

test_maks  = set(test_df['MAK'].tolist())
train_maks = train_df['MAK'].values

# ============================================================
# STEP 6: Normalize Consumer Features to 0-1 Scale
# ============================================================
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(train_df[consumer_feature_cols])
X_test  = scaler.transform(test_df[consumer_feature_cols])
X_all   = scaler.transform(consumer_clean[consumer_feature_cols])

# ============================================================
# STEP 7: Train KNN
# Finds K most similar consumers from training set
# Recommends the properties those consumers actually chose
# ============================================================
K = 10
print(f'\nTraining KNN (K={K}, metric=euclidean)...')
knn = NearestNeighbors(n_neighbors=K, metric='euclidean', n_jobs=-1)
knn.fit(X_train)
print('Done.')

# ============================================================
# STEP 8: Validate on Test Set
# For each test consumer, did we recommend their actual house?
# ============================================================
print('\n' + '='*60)
print('VALIDATION ON TEST SET (20%)')
print('='*60)

actual_map = dict(zip(prop['MAK'], prop['PropertyAddress']))

hit_top1  = 0
hit_top5  = 0
hit_top10 = 0
total     = len(test_df)

for i, (_, test_row) in enumerate(test_df.iterrows()):
    actual_address = actual_map.get(test_row['MAK'])
    if not actual_address:
        continue

    test_vec           = X_test[i].reshape(1, -1)
    distances, indices = knn.kneighbors(test_vec)
    neighbor_maks      = train_maks[indices[0]]
    neighbor_addresses = [actual_map.get(m, '') for m in neighbor_maks]

    if actual_address in neighbor_addresses[:1]:  hit_top1  += 1
    if actual_address in neighbor_addresses[:5]:  hit_top5  += 1
    if actual_address in neighbor_addresses[:10]: hit_top10 += 1

print(f'\n  Test consumers:          {total}')
print(f'  Actual house in Top 1:   {hit_top1}  ({hit_top1/total*100:.1f}%)')
print(f'  Actual house in Top 5:   {hit_top5}  ({hit_top5/total*100:.1f}%)')
print(f'  Actual house in Top 10:  {hit_top10} ({hit_top10/total*100:.1f}%)')

# ============================================================
# STEP 9: Generate Top 10 Recommendations for ALL Consumers
# ============================================================
print('\nGenerating recommendations for all consumers...')

distances_all, indices_all = knn.kneighbors(X_all)
results = []

for i in range(len(consumer_clean)):
    mak_val  = consumer_clean['MAK'].iloc[i]
    dists    = distances_all[i]
    idxs     = indices_all[i]
    max_dist = dists.max() if dists.max() > 0 else 1

    seen = []
    rank = 1

    for dist, idx in zip(dists, idxs):
        neighbor_mak     = train_maks[idx]
        neighbor_address = actual_map.get(neighbor_mak)
        if not neighbor_address or neighbor_address in seen:
            continue

        prop_row = p[p['MAK'] == neighbor_mak]
        if prop_row.empty:
            continue

        prop_row    = prop_row.iloc[0]
        match_score = round(1 - (dist / max_dist), 4)
        seen.append(neighbor_address)

        results.append({
            'ConsumerMAK':               mak_val,
            'IsTestConsumer':            mak_val in test_maks,
            'Rank':                      rank,
            'PropertyMAK':               neighbor_mak,
            'PropertyAddress':           neighbor_address,
            'NumberOfBedrooms':          prop_row['NumberOfBedrooms'],
            'NumberOfBaths':             prop_row['NumberOfBaths'],
            'LotSizeOrArea':             prop_row['LotSizeOrArea'],
            'TotalAssessedValue':        prop_row['TotalAssessedValue'],
            'SalesPriceFromAssessment':  prop_row['SalesPriceFromAssessment'],
            'YearBuilt':                 prop_row['YearBuilt'],
            'GarageParkingNumberOfCars': prop_row['GarageParkingNumberOfCars'],
            'HasPool':                   prop_row['HasPool'],
            'MatchScore':                match_score,
        })
        rank += 1
        if rank > 10:
            break

results_df = pd.DataFrame(results)
print(f'Generated {len(results_df):,} recommendations.')

# ============================================================
# STEP 10: Inspect Specific Consumer
# Change this MAK to whoever you want to inspect
# ============================================================
inspect_mak  = 1937927176
inspect_recs = results_df[results_df['ConsumerMAK'] == inspect_mak]

if not inspect_recs.empty:
    actual_addr = actual_map.get(inspect_mak, 'Not in property data')
    print(f'\nInspecting Consumer MAK: {inspect_mak}')
    print(f'Their actual address:    {actual_addr}')
    print(f'\nTop 10 recommendations:')
    print(inspect_recs.to_string(index=False))
else:
    print(f'\nMAK {inspect_mak} not found.')

# ============================================================
# STEP 11: Save
# ============================================================
results_df.to_csv('property_recommendations.csv', index=False)
print('\nSaved: property_recommendations.csv')
