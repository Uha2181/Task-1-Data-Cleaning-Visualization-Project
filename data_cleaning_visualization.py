# ============================================================
# Task 1: Data Cleaning & Visualization Project
# Dataset: Titanic (loaded via seaborn)
# Author: Data Science Intern
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = sns.load_dataset('titanic')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn names: {list(df.columns)}")

# ─────────────────────────────────────────────
# STEP 2: INITIAL EXPLORATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Initial Data Exploration")
print("=" * 60)

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values (before cleaning):")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# ─────────────────────────────────────────────
# STEP 3: HANDLE MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Handling Missing Values")
print("=" * 60)

# Fill 'age' with median
age_median = df['age'].median()
df['age'].fillna(age_median, inplace=True)
print(f"✔ Filled 'age' missing values with median: {age_median}")

# Fill 'embarked' with mode
embarked_mode = df['embarked'].mode()[0]
df['embarked'].fillna(embarked_mode, inplace=True)
print(f"✔ Filled 'embarked' missing values with mode: {embarked_mode}")

# Drop 'deck' column (too many missing values >77%)
df.drop(columns=['deck'], inplace=True)
print("✔ Dropped 'deck' column (>77% missing)")

# Fill 'embark_town' with mode
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)
print("✔ Filled 'embark_town' missing values with mode")

print("\nMissing Values (after cleaning):")
print(df.isnull().sum())

# ─────────────────────────────────────────────
# STEP 4: HANDLE DUPLICATES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Handling Duplicates")
print("=" * 60)

duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"✔ Removed {duplicates} duplicate rows")
else:
    print("✔ No duplicate rows found")

# ─────────────────────────────────────────────
# STEP 5: HANDLE OUTLIERS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Handling Outliers (IQR Method)")
print("=" * 60)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    after = len(df)
    print(f"✔ '{column}': removed {before - after} outliers (range: {lower:.2f} – {upper:.2f})")
    return df

df = remove_outliers_iqr(df, 'age')
df = remove_outliers_iqr(df, 'fare')

print(f"\nDataset shape after cleaning: {df.shape}")

# ─────────────────────────────────────────────
# STEP 6: FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Feature Engineering")
print("=" * 60)

df['family_size'] = df['sibsp'] + df['parch'] + 1
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                          labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
print("✔ Created 'family_size' feature")
print("✔ Created 'age_group' feature")

# ─────────────────────────────────────────────
# STEP 7: VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Creating Visualizations")
print("=" * 60)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.facecolor'] = '#f8f9fa'

# ── Figure 1: Overview Dashboard ──
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Titanic Dataset — Data Analysis Dashboard', 
             fontsize=22, fontweight='bold', y=0.98, color='#2c3e50')

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Survival Count
ax1 = fig.add_subplot(gs[0, 0])
survival_counts = df['survived'].value_counts()
bars = ax1.bar(['Did Not Survive', 'Survived'], survival_counts.values,
               color=['#e74c3c', '#2ecc71'], edgecolor='white', linewidth=1.5)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             str(int(bar.get_height())), ha='center', va='bottom', fontweight='bold')
ax1.set_title('Survival Count', fontweight='bold', fontsize=12)
ax1.set_ylabel('Number of Passengers')

# Plot 2: Survival by Gender
ax2 = fig.add_subplot(gs[0, 1])
gender_survival = df.groupby('sex')['survived'].mean() * 100
bars2 = ax2.bar(gender_survival.index, gender_survival.values,
                color=['#3498db', '#e91e8c'], edgecolor='white', linewidth=1.5)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')
ax2.set_title('Survival Rate by Gender', fontweight='bold', fontsize=12)
ax2.set_ylabel('Survival Rate (%)')
ax2.set_ylim(0, 100)

# Plot 3: Survival by Class
ax3 = fig.add_subplot(gs[0, 2])
class_survival = df.groupby('pclass')['survived'].mean() * 100
bars3 = ax3.bar(['1st Class', '2nd Class', '3rd Class'], class_survival.values,
                color=['#f39c12', '#9b59b6', '#1abc9c'], edgecolor='white', linewidth=1.5)
for bar in bars3:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')
ax3.set_title('Survival Rate by Class', fontweight='bold', fontsize=12)
ax3.set_ylabel('Survival Rate (%)')
ax3.set_ylim(0, 100)

# Plot 4: Age Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df[df['survived']==0]['age'], bins=25, alpha=0.7, color='#e74c3c', label='Did Not Survive')
ax4.hist(df[df['survived']==1]['age'], bins=25, alpha=0.7, color='#2ecc71', label='Survived')
ax4.set_title('Age Distribution by Survival', fontweight='bold', fontsize=12)
ax4.set_xlabel('Age')
ax4.set_ylabel('Count')
ax4.legend()

# Plot 5: Fare Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df['fare'], bins=30, color='#3498db', edgecolor='white', linewidth=0.8)
ax5.axvline(df['fare'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['fare'].mean():.1f}")
ax5.axvline(df['fare'].median(), color='orange', linestyle='--', linewidth=2, label=f"Median: {df['fare'].median():.1f}")
ax5.set_title('Fare Distribution', fontweight='bold', fontsize=12)
ax5.set_xlabel('Fare (£)')
ax5.set_ylabel('Count')
ax5.legend()

# Plot 6: Embarked Port
ax6 = fig.add_subplot(gs[1, 2])
port_counts = df['embarked'].value_counts()
wedges, texts, autotexts = ax6.pie(port_counts.values, labels=port_counts.index,
                                    autopct='%1.1f%%', colors=['#3498db', '#e74c3c', '#2ecc71'],
                                    startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
ax6.set_title('Embarked Port Distribution', fontweight='bold', fontsize=12)

# Plot 7: Age Group Survival
ax7 = fig.add_subplot(gs[2, 0])
age_surv = df.groupby('age_group', observed=True)['survived'].mean() * 100
ax7.bar(age_surv.index, age_surv.values,
        color=['#ff6b6b','#feca57','#48dbfb','#ff9ff3','#54a0ff'],
        edgecolor='white', linewidth=1.5)
ax7.set_title('Survival Rate by Age Group', fontweight='bold', fontsize=12)
ax7.set_ylabel('Survival Rate (%)')
ax7.set_xlabel('Age Group')

# Plot 8: Family Size vs Survival
ax8 = fig.add_subplot(gs[2, 1])
fam_surv = df.groupby('family_size')['survived'].mean() * 100
ax8.plot(fam_surv.index, fam_surv.values, marker='o', linewidth=2.5,
         color='#6c5ce7', markersize=8, markerfacecolor='white', markeredgewidth=2)
ax8.fill_between(fam_surv.index, fam_surv.values, alpha=0.2, color='#6c5ce7')
ax8.set_title('Survival Rate by Family Size', fontweight='bold', fontsize=12)
ax8.set_xlabel('Family Size')
ax8.set_ylabel('Survival Rate (%)')

# Plot 9: Heatmap - Class vs Gender
ax9 = fig.add_subplot(gs[2, 2])
pivot = df.pivot_table(values='survived', index='sex', columns='pclass', aggfunc='mean') * 100
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
            ax=ax9, linewidths=0.5, cbar_kws={'label': 'Survival %'})
ax9.set_title('Survival % (Gender × Class)', fontweight='bold', fontsize=12)
ax9.set_xlabel('Passenger Class')
ax9.set_ylabel('Gender')

plt.savefig('/mnt/user-data/outputs/titanic_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#f8f9fa')
print("✔ Saved: titanic_dashboard.png")
plt.close()

# ── Figure 2: Correlation Heatmap ──
fig2, ax = plt.subplots(figsize=(10, 7))
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix — Numeric Features', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("✔ Saved: correlation_heatmap.png")
plt.close()

# ─────────────────────────────────────────────
# STEP 8: SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Final Summary Report")
print("=" * 60)

total = len(df)
survived = df['survived'].sum()
print(f"\n📊 FINAL DATASET SUMMARY")
print(f"   Total Passengers (after cleaning): {total}")
print(f"   Survived: {survived} ({survived/total*100:.1f}%)")
print(f"   Did Not Survive: {total-survived} ({(total-survived)/total*100:.1f}%)")
print(f"\n👤 GENDER ANALYSIS")
for gender in df['sex'].unique():
    subset = df[df['sex']==gender]
    rate = subset['survived'].mean()*100
    print(f"   {gender.title()}: {rate:.1f}% survival rate ({len(subset)} passengers)")
print(f"\n🚢 CLASS ANALYSIS")
for cls in sorted(df['pclass'].unique()):
    subset = df[df['pclass']==cls]
    rate = subset['survived'].mean()*100
    print(f"   Class {cls}: {rate:.1f}% survival rate ({len(subset)} passengers)")
print(f"\n📈 AGE STATS")
print(f"   Mean Age: {df['age'].mean():.1f} years")
print(f"   Median Age: {df['age'].median():.1f} years")
print(f"   Age Range: {df['age'].min():.0f} – {df['age'].max():.0f} years")
print(f"\n💷 FARE STATS")
print(f"   Mean Fare: £{df['fare'].mean():.2f}")
print(f"   Median Fare: £{df['fare'].median():.2f}")

print("\n✅ ALL TASKS COMPLETE — Charts saved to output folder!")
