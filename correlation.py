import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

base_path = "./dataset/"

channels = pd.read_csv(base_path + "channels.csv", encoding='latin-1')
deliveries = pd.read_csv(base_path + "deliveries.csv", encoding='latin-1')
drivers = pd.read_csv(base_path + "drivers.csv", encoding='latin-1')
hubs = pd.read_csv(base_path + "hubs.csv", encoding='latin-1')
orders = pd.read_csv(base_path + "orders.csv", encoding='latin-1')
payments = pd.read_csv(base_path + "payments.csv", encoding='latin-1')
stores = pd.read_csv(base_path + "stores.csv", encoding='latin-1')

# Merge deliveries com orders usando delivery_order_id
df = deliveries.merge(orders, left_on="delivery_order_id", right_on="order_id", how="left")
df = df.merge(channels, on="channel_id", how="left")
df = df.merge(drivers, on="driver_id", how="left")
df = df.merge(stores, on="store_id", how="left")
df = df.merge(hubs, on="hub_id", how="left")
df = df.merge(payments, left_on="order_id", right_on="payment_order_id", how="left")

# Initial exploration
print("Formato final do dataset:", df.shape)
print("Colunas:", df.columns.tolist())
print(df.head())

# Target - usar delivery_status ou order_status
df['target'] = df['delivery_status'].map({
    'DELIVERED': 1,
    'CANCELLED': 0
})

df = df.dropna(subset=['target'])

# Remover colunas de ID que não agregam informação útil para correlação
id_columns = [
    'delivery_id', 'delivery_order_id', 'order_id', 'payment_order_id',
    'payment_id', 'driver_id', 'channel_id', 'store_id', 'hub_id',
    'delivery_order_id_x', 'delivery_order_id_y',
    'payment_order_id_x', 'payment_order_id_y'
]

# Selecionar apenas colunas numéricas e remover IDs
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
features_for_corr = [col for col in numeric_cols if col not in id_columns]

# Manter o target se existir
if 'target' in df.columns:
    features_for_corr.append('target')

print(f"\nColunas usadas para correlação ({len(features_for_corr)}):")
print(features_for_corr)

plt.figure(figsize=(12, 10))
correlation = df[features_for_corr].corr()

# Filtrar apenas correlações relevantes (opcional: remover correlações muito baixas)
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)

plt.title('Matriz de Correlação (sem colunas de ID)', fontsize=14, pad=20)
plt.tight_layout()
plt.show()
