import pandas as pd
import os

data_dir = 'dataset'

def load_data(filename):
    path = os.path.join(data_dir, filename)
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1')

channels = load_data('channels.csv')
deliveries = load_data('deliveries.csv')
drivers = load_data('drivers.csv')
hubs = load_data('hubs.csv')
orders = load_data('orders.csv')
stores = load_data('stores.csv')

df = deliveries.merge(orders, left_on='delivery_order_id', right_on='order_id', how='left')
df = df.merge(channels, on='channel_id', how='left')
df = df.merge(stores, on='store_id', how='left')
df = df.merge(hubs, on='hub_id', how='left')
df = df.merge(drivers, on='driver_id', how='left')

print(f"Merged shape: {df.shape}")
print("Columns:", df.columns.tolist())

print("\nTarget Distribution:")
print(df['delivery_status'].value_counts(normalize=True))

print("\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

df = df.dropna(subset=['delivery_status'])

output_path = os.path.join(data_dir, 'merged.csv')
df.to_csv(output_path, index=False)
print(f"\nSaved merged data to {output_path}")
