import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style="whitegrid") 


def get_post_split_volume(row):
    try:
        ticker = row['Ticker']
        event_date = pd.to_datetime(row['Date'])
        start = event_date + timedelta(days=1)
        end = event_date + timedelta(days=45) 
        
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty: return np.nan
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        if 'Volume' in data.columns:
            avg_vol = data['Volume'].iloc[:30].mean()
            if isinstance(avg_vol, (pd.Series, np.ndarray, list)):
                if hasattr(avg_vol, 'iloc'): return float(avg_vol.iloc[0])
                elif len(avg_vol) > 0: return float(avg_vol[0])
            return float(avg_vol)
        return np.nan
    except: return np.nan

def get_trend_data(row):
    try:
        ticker = row['Ticker']
        event_date = pd.to_datetime(row['Date'])
        
        # Window: 30 days before to 30 days after
        start = event_date - timedelta(days=45)
        end = event_date + timedelta(days=45)
        
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty: return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data['Relative_Day'] = (data.index - event_date).days
        data = data[(data['Relative_Day'] >= -30) & (data['Relative_Day'] <= 30)]
        
        data['Log_Volume'] = np.log(data['Volume'])
        
        return data[['Relative_Day', 'Log_Volume']]
    except: return None

print("Loading dataset...")
try:
    df = pd.read_csv("causal_dataset_large.csv")
except FileNotFoundError:
    print("Error: 'causal_dataset_large.csv' not found.")
    exit()

print("Estimating Propensity Scores...")
df['Log_Price'] = np.log(df['Price'])
df['Log_Vol_30d'] = np.log(df['Avg_Volume_30d'])
covariates = ['Momentum_6m', 'Volatility_30d', 'Log_Price', 'Log_Vol_30d']

logit = LogisticRegression(solver='liblinear')
logit.fit(df[covariates], df['Treated'])
df['pscore'] = logit.predict_proba(df[covariates])[:, 1]

print("Matching...")
treated = df[df['Treated'] == 1]
control = df[df['Treated'] == 0]

nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(control[['pscore']])
distances, indices = nn.kneighbors(treated[['pscore']])

matched_controls = control.iloc[indices.flatten()]
matched_df = pd.concat([treated, matched_controls])
print(f"Matched Dataset Size: {len(matched_df)}")


print("\nGenerating Plots...")

plt.figure(figsize=(10, 6))
sns.kdeplot(treated['pscore'], fill=True, label='Treated (Pre-Match)', color='blue', alpha=0.3)
sns.kdeplot(control['pscore'], fill=True, label='Control (Pre-Match)', color='red', alpha=0.3)
plt.title('Propensity Score Distribution (Before Matching)')
plt.xlabel('Probability of Stock Split')
plt.legend()
plt.savefig('1_pscore_overlap.png')
print("-> Saved '1_pscore_overlap.png'")

# PLOT 2: Covariate Balance (Love Plot)
def calculate_smd(df_in, treatment_col='Treated', covariates=covariates):
    treated = df_in[df_in[treatment_col] == 1]
    control = df_in[df_in[treatment_col] == 0]
    
    smds = []
    for cov in covariates:
        mean_t = treated[cov].mean()
        mean_c = control[cov].mean()
        std_pool = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
        smd = (mean_t - mean_c) / std_pool
        smds.append(abs(smd)) # Absolute SMD
    return smds

smd_pre = calculate_smd(df)
smd_post = calculate_smd(matched_df)

balance_df = pd.DataFrame({
    'Covariate': covariates,
    'Unmatched': smd_pre,
    'Matched': smd_post
})
# Convert to long format for plotting
balance_long = pd.melt(balance_df, id_vars='Covariate', var_name='Dataset', value_name='Abs_SMD')

plt.figure(figsize=(8, 5))
sns.scatterplot(data=balance_long, x='Abs_SMD', y='Covariate', hue='Dataset', s=100)
plt.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.1)')
plt.title('Covariate Balance (Love Plot)')
plt.xlabel('Absolute Standardized Mean Difference')
plt.legend()
plt.tight_layout()
plt.savefig('2_covariate_balance.png')
print("-> Saved '2_covariate_balance.png'")

# PLOT 3: Parallel Trends (DiD Visual)
print("Fetching daily history for Parallel Trends plot (this takes time)...")
trend_data = []

for idx, row in matched_df.iterrows():
    data = get_trend_data(row)
    if data is not None:
        data['Treated_Group'] = "Treated" if row['Treated'] == 1 else "Control"
        trend_data.append(data)

if trend_data:
    trend_df = pd.concat(trend_data)
    
    # Aggregate by relative day and group
    summary = trend_df.groupby(['Relative_Day', 'Treated_Group'])['Log_Volume'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=summary, x='Relative_Day', y='Log_Volume', hue='Treated_Group', style='Treated_Group', linewidth=2.5)
    
    # Add vertical line for Split Date
    plt.axvline(x=0, color='black', linestyle='--', label='Split Date')
    plt.axvspan(0, 30, color='gray', alpha=0.1, label='Post-Treatment Window')
    
    plt.title('Parallel Trends Check: Average Log Volume Around Split Date')
    plt.xlabel('Days Relative to Split (0 = Split Day)')
    plt.ylabel('Average Log Volume')
    plt.legend()
    plt.savefig('3_parallel_trends.png')
    print("-> Saved '3_parallel_trends.png'")

# --- 6. REGRESSION ANALYSIS ---
print("\nRunning Final DiD Regression...")
matched_df['Post_Volume'] = matched_df.apply(get_post_split_volume, axis=1)
matched_df.dropna(subset=['Post_Volume'], inplace=True)

matched_df['Log_Post_Vol'] = np.log(matched_df['Post_Volume'])
matched_df['Log_Pre_Vol']  = matched_df['Log_Vol_30d'] 
matched_df['Volume_Change'] = matched_df['Log_Post_Vol'] - matched_df['Log_Pre_Vol']

model = smf.ols("Volume_Change ~ Treated", data=matched_df)
results = model.fit()

print("\n" + "="*40)
print("       FINAL CAUSAL RESULTS       ")
print("="*40)
print(results.summary())


print("\n" + "="*40)
print("    EXTENSION 1: HETEROGENEITY (HTE)    ")
print("="*40)


hte_model = smf.ols("Volume_Change ~ Treated * Momentum_6m + Treated * Volatility_30d", data=matched_df)
hte_results = hte_model.fit()

print(hte_results.summary())

print("--- Interpretation ---")
interaction_coeff = hte_results.params.get('Treated:Momentum_6m', 0)
pval = hte_results.pvalues.get('Treated:Momentum_6m', 1.0)

if pval < 0.1:
    print(f"Result: The interaction is significant (p={pval:.3f}).")
    if interaction_coeff < 0:
        print("Insight: Firms with HIGH momentum get a SMALLER boost from splitting.")
        print("This supports the 'Signaling' hypothesis (good news was already priced in).")
    else:
        print("Insight: Firms with HIGH momentum get a LARGER boost from splitting.")
else:
    print("Result: No significant interaction. The split effect appears consistent across different momentum levels.")


print("\n" + "="*40)
print("    EXTENSION 2: FISHER'S PERMUTATION TEST   ")
print("="*40)
print("Running 1,000 simulations to validate robustness (this takes ~10 seconds)...")

actual_effect = results.params['Treated']

# 2. Run Simulations
fake_effects = []
n_simulations = 1000

temp_df = matched_df.copy()

for _ in range(n_simulations):
    # Shuffle the treatment assignment column
    temp_df['Fake_Treated'] = np.random.permutation(temp_df['Treated'].values)
    
    # Run the exact same regression on the shuffled data
    fake_model = smf.ols("Volume_Change ~ Fake_Treated", data=temp_df)
    res = fake_model.fit()
    fake_effects.append(res.params['Fake_Treated'])

# 3. Calculate Empirical P-Value
fake_effects = np.array(fake_effects)
# How often was the fake effect bigger than our real effect?
p_value_perm = (np.abs(fake_effects) >= np.abs(actual_effect)).mean()

# 4. Plot Results
plt.figure(figsize=(10, 6))
plt.hist(fake_effects, bins=30, alpha=0.7, color='gray', label='Random Chance (Placebo)')
plt.axvline(x=actual_effect, color='red', linestyle='--', linewidth=2, label=f'Actual Effect ({actual_effect:.3f})')
plt.title(f'Fisher Permutation Test (p = {p_value_perm:.3f})')
plt.xlabel('Estimated Causal Effect')
plt.legend()
plt.savefig('4_permutation_test.png')
print("-> Saved '4_permutation_test.png'")

print(f"\nEmpirical P-Value: {p_value_perm:.4f}")
if p_value_perm < 0.05:
    print("CONCLUSION: Your result is ROBUST. It is extremely unlikely to happen by chance.")
else:
    print("CONCLUSION: Your result might be noise. The permutation test failed to reject the null.")