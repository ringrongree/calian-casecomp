import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

customers = pd.read_csv("datasets/customers.csv")
orders = pd.read_csv("datasets/orders.csv")
details = pd.read_csv("datasets/order_details.csv")
loyalty = pd.read_csv("datasets/loyalty_members.csv")
foods = pd.read_csv("datasets/food_menu.csv")
games = pd.read_csv("datasets/games_menu.csv")
drinks = pd.read_csv("datasets/drinks_menu.csv")

orders['transaction_date'] = pd.to_datetime(orders['transaction_date'])
customers['loyalty_enrolled'] = (customers['loyalty_enrolled'] == 'Y').astype(int)
details['is_food'] = (details['item_type'] == 'food').astype(int)
details['is_drink'] = (details['item_type'] == 'drink').astype(int)
details['is_game'] = (details['item_type'] == 'entertainment').astype(int)


tx_totals = (details
             .groupby('transaction_id', as_index=False)
             .agg(items_qty=('quantity','sum')))

orders = orders.merge(tx_totals, on='transaction_id', how='left')

tx_food = details.groupby('transaction_id')['line_amount_cad'].apply(
    lambda x: details.loc[x.index, 'is_food'].mul(details.loc[x.index, 'line_amount_cad']).sum()
).rename('food_spend')

tx_game = details.groupby('transaction_id')['line_amount_cad'].apply(
    lambda x: details.loc[x.index, 'is_game'].mul(details.loc[x.index, 'line_amount_cad']).sum()
).rename('game_spend')

tx_drink = details.groupby('transaction_id')['line_amount_cad'].apply(
    lambda x: details.loc[x.index, 'is_drink'].mul(details.loc[x.index, 'line_amount_cad']).sum()
).rename('drink_spend')


orders = orders.merge(tx_food, on='transaction_id', how='left')
orders = orders.merge(tx_game, on='transaction_id', how='left')
orders = orders.merge(tx_drink, on='transaction_id', how='left')
orders[['food_spend','game_spend', 'drink_spend']] = orders[['food_spend','game_spend', 'drink_spend']].fillna(0)
orders = orders.sort_values(by=['email', 'transaction_date'])
orders['visit_number'] = orders.groupby('email').cumcount() + 1
order_cust = orders.merge(customers, on='email', how='left')

fv = order_cust[order_cust['visit_number'] == 1].copy()

fv['total_spend'] = fv['total_bill_amount']
fv['food_drink_ratio'] = (fv['food_spend']+fv['drink_spend']) / fv['total_spend'].replace(0, np.nan)
fv['game_ratio'] = fv['game_spend'] / fv['total_spend'].replace(0, np.nan)

fv[['food_drink_ratio','game_ratio']] = fv[['food_drink_ratio','game_ratio']].fillna(0)
fv['items_per_min'] = fv['items_qty'] / fv['time_spent_min'].replace(0, np.nan)

fv['spend_per_min'] = fv['total_spend'] / fv['time_spent_min'].replace(0, np.nan)

fv[['spend_per_min']] = fv[['spend_per_min']].fillna(0)

cluster_features = fv[[
    'food_drink_ratio','game_ratio',
    'total_spend','items_qty','time_spent_min',
    'items_per_min','spend_per_min'
]]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
fv['cluster'] = kmeans.fit_predict(X_scaled)

numeric_cols = ['food_drink_ratio','game_ratio','total_spend']
cluster_map = fv.groupby('cluster')[numeric_cols].mean()
count = fv.groupby('cluster')[numeric_cols].count()
print(cluster_map)
print(count)


fv['persona'] = 'Undefined'

fv.loc[
    (fv['game_ratio'] > 0.4) &
    (fv['total_spend'] > fv['total_spend'].quantile(0.25)),
    'persona'
] = 'Gamers'

# Food-First (high ratio AND meaningful food or drink spend)
fv.loc[
    (fv['food_drink_ratio'] > 0.8) &
    (fv['total_spend'] > fv['total_spend'].quantile(0.25)),
    'persona'
] = 'Foodies'

# Balanced (meaningful spend across both)
fv.loc[
   (fv['game_ratio'].between(0.3,0.7)) &
   (fv['food_drink_ratio'].between(0.3,0.7)) &
   (fv['total_spend'] > fv['total_spend'].quantile(0.25)),
   'persona'
] = 'All-Rounders'


fv.loc[
   (fv['total_spend'] < fv['total_spend'].quantile(0.25)) &
   (fv['items_qty'] < fv['items_qty'].quantile(0.25)),
   'persona'
] = 'Drop-Ins'

fv['persona'] = fv['persona'].replace('Undefined','All-Rounders')

print(fv['persona'].value_counts())
print(fv.groupby('persona')[['total_spend','food_spend','drink_spend','game_spend','time_spent_min']].mean())

persona_dollars = fv.groupby('persona')[['food_spend','drink_spend','game_spend']].mean()

print(persona_dollars.round(2))


persona_order = ['Foodies', 'All-Rounders', 'Gamers', 'Drop-Ins']
persona_dollars = persona_dollars.reindex(persona_order)

persona_dollars.plot(kind='bar', stacked=True, figsize=(8,5),
                     color=['#084266', '#6eb5e5', '#c0e3f9'],  # food, drink, game
                     edgecolor='black')

plt.title("Average Spend Breakdown by Persona (Dollars, First Visit)")
plt.ylabel("Average Spend ($)")
plt.xlabel("Persona")
plt.xticks(rotation=0, ha='center')
plt.legend(['Food Spend', 'Drink Spend', 'Game Spend'], title="Spend Category", bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.savefig("plot1.png", dpi=300)
plt.close()

# average number of visits per customer by persona
visit_counts = orders.groupby('email')['visit_number'].max().reset_index()
visit_counts.columns = ['email', 'total_visits']
fv = fv.merge(visit_counts, on='email', how='left')
avg_visits = fv.groupby('persona')['total_visits'].mean().round(2)
print(avg_visits)

# average number of visits per customer by persona
spend = orders.groupby('email')['total_bill_amount'].sum().reset_index()
spend.columns = ['email', 'total_spend']
fv = fv.merge(spend, on='email', how='left')
total_spend = fv.groupby('persona')['total_bill_amount'].mean().round(2)
print(total_spend)

# Churn analysis
latest_date = orders['transaction_date'].max()
last_visit = orders.groupby('email')['transaction_date'].max().reset_index()
last_visit['days_since_last_visit'] = (latest_date - last_visit['transaction_date']).dt.days
last_visit['churned'] = (last_visit['days_since_last_visit'] > 90).astype(int)
fv = fv.merge(last_visit[['email','days_since_last_visit','churned']], on='email', how='left')
print(fv.groupby('persona')['churned'].mean().round(3))




orders['food_drink_ratio'] = (orders['food_spend'] + orders['drink_spend']) / orders['total_bill_amount'].replace(0,
                                                                                                                  np.nan)
orders['game_ratio'] = orders['game_spend'] / orders['total_bill_amount'].replace(0, np.nan)
orders[['food_drink_ratio', 'game_ratio']] = orders[['food_drink_ratio', 'game_ratio']].fillna(0)

orders['persona_visit'] = 'Undefined'

orders.loc[
    (orders['game_ratio'] > 0.4) &
    (orders['game_spend'] > orders['game_spend'].quantile(0.25)),
    'persona_visit'
] = 'Gamers'

orders.loc[
    (orders['food_drink_ratio'] > 0.8) &
    (
        (orders['food_spend'] > orders['food_spend'].quantile(0.25)) |
        (orders['drink_spend'] > orders['drink_spend'].quantile(0.25))
    ),
    'persona_visit'
] = 'Foodies'

orders.loc[
    (orders['game_ratio'].between(0.3, 0.7)) &
    (orders['food_drink_ratio'].between(0.3, 0.7)) &
    (orders['total_bill_amount'] > orders['total_bill_amount'].quantile(0.25)),
    'persona_visit'
] = 'All-Rounders'

orders.loc[
    (orders['total_bill_amount'] < orders['total_bill_amount'].quantile(0.25)) &
    (orders['items_qty'] < orders['items_qty'].quantile(0.25)),
    'persona_visit'
] = 'Drop-Ins'

orders['persona_visit'] = orders['persona_visit'].replace('Undefined', 'All-Rounders')

orders = orders.merge(fv[['email', 'persona']], on='email', how='left', suffixes=('', '_first'))
orders = orders.rename(columns={'persona': 'persona_first'})


transitions_by_visit_counts = {}
max_visit = orders['visit_number'].max()

for v in range(1, max_visit):
    current = orders[orders['visit_number'] == v][['email', 'persona_visit']]
    next_v = orders[orders['visit_number'] == v + 1][['email', 'persona_visit']]

    cust_current = current['email'].nunique()
    cust_next = next_v['email'].nunique()

    df = current.merge(next_v, on='email', suffixes=(f'_v{v}', f'_v{v + 1}'))
    if df.empty:
        continue

    trans_counts = (
        df.groupby([f'persona_visit_v{v}', f'persona_visit_v{v + 1}'])
        .size().unstack().fillna(0).astype(int)
    )

    transitions_by_visit_counts[f'{v}->{v + 1}'] = trans_counts

    print(f"\n=== Persona Transition Counts: Visit {v} -> Visit {v + 1} ===")
    print(f"Customers who reached Visit {v}: {cust_current}")
    print(f"Customers who reached Visit {v + 1}: {cust_next}")
    print(trans_counts)

#Behavior change drivers

first_visit_persona = orders[orders['visit_number'] == 1][['email', 'persona_visit']]
first_visit_persona = first_visit_persona.rename(columns={'persona_visit': 'persona_v1'})

low_first = first_visit_persona[first_visit_persona['persona_v1'] == 'Drop-Ins']

became_bal = orders[orders['persona_visit'] == 'All-Rounders'][['email']].drop_duplicates()


upgrade_customers = low_first.merge(became_bal, on='email', how='inner')

print(f"\nTotal Low → Balanced Upgraders: {len(upgrade_customers)}")

upgrade_orders = orders.merge(upgrade_customers[['email']], on='email')

v1 = upgrade_orders[upgrade_orders['visit_number'] == 1]
v2 = upgrade_orders[upgrade_orders['visit_number'] == 2]

compare = v1[['email','total_bill_amount','time_spent_min','game_spend','food_spend','drink_spend']] \
    .merge(v2[['email','total_bill_amount','time_spent_min','game_spend','food_spend','drink_spend']],
           on='email', suffixes=('_v1','_v2'))

compare['spend_per_min_v1'] = compare['total_bill_amount_v1'] / compare['time_spent_min_v1'].replace(0,np.nan)
compare['spend_per_min_v2'] = compare['total_bill_amount_v2'] / compare['time_spent_min_v2'].replace(0,np.nan)

compare['delta_spend'] = compare['total_bill_amount_v2'] - compare['total_bill_amount_v1']
compare['delta_spend_per_min'] = compare['spend_per_min_v2'] - compare['spend_per_min_v1']
compare['delta_game_spend'] = compare['game_spend_v2'] - compare['game_spend_v1']
compare['delta_food_spend'] = compare['food_spend_v2'] - compare['food_spend_v1']
compare['delta_drink_spend'] = compare['drink_spend_v2'] - compare['drink_spend_v1']
compare['delta_time_spent'] = compare['time_spent_min_v2'] - compare['time_spent_min_v1']

print("\nAverage behavior change for customers upgrading Low → Balanced:")
print(compare[['delta_spend', 'delta_spend_per_min', 'delta_game_spend', 'delta_time_spent','delta_food_spend', 'delta_drink_spend']].mean().round(2))

# Menu and Basket insight
detail_persona = details.merge(orders[['transaction_id','persona_visit']], on='transaction_id')

top_items = (
    detail_persona.groupby(['persona_visit','item_name'])
    .size().reset_index(name='count')
    .sort_values(['persona_visit','count'], ascending=[True,False])
)

for persona in top_items['persona_visit'].unique():
    print(f"\n--- {persona} ---")
    print(top_items[top_items['persona_visit']==persona].head(10))

details['is_alcohol'] = details['item_name'].str.contains(
    'Beer|Wine|Margarita|Old Fashioned|Cocktail|Whiskey|Sangria',
    case=False, regex=True
).astype(int)

tx_alcohol = details.groupby('transaction_id')['line_amount_cad'].apply(
    lambda x: details.loc[x.index, 'is_alcohol'].mul(details.loc[x.index, 'line_amount_cad']).sum()
).rename('alcohol_spend')

orders = orders.merge(tx_alcohol, on='transaction_id', how='left').fillna({'alcohol_spend':0})
print(orders.groupby('persona_visit')['alcohol_spend'].mean()
)


for v in range(1, min(max_visit, 5)):
    if f"{v}->{v + 1}" in transitions_by_visit_counts:
        transition = transitions_by_visit_counts[f"{v}->{v + 1}"]
        if transition.size == 0:
            continue

        plt.figure(figsize=(6, 5))
        sns.heatmap(transition, annot=True, cmap="Blues", fmt="d")
        plt.title(f"Persona Transition Heatmap (Visit {v} → {v + 1})")
        plt.xlabel(f"Visit {v + 1}")
        plt.ylabel(f"Visit {v}")
        plt.tight_layout()
        plt.savefig(f"transition_v{v}_to_v{v + 1}.png", dpi=300)
        plt.close()

