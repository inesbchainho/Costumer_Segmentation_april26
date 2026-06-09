import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_customer_info(df):
    """
    Cleans customer_info and creates additional features for customer segmentation.
    """
    df_clean = df.copy()

    def extract_degree(name):
        match = re.match(r'^(Bsc|Msc|Phd)\.', str(name))
        return match.group(1) if match else 'None'

    df_clean['degree'] = df_clean['customer_name'].apply(extract_degree)
    df_clean['degree_num'] = df_clean["degree"].map({'Bsc': 1, 'Msc': 2, 'Phd': 3, 'None': 0})

    df_clean["is_female"] = (df_clean["customer_gender"] == "female").astype(int)
    
    df_clean["customer_birthdate"] = pd.to_datetime(df_clean["customer_birthdate"], format = "%m/%d/%Y %I:%M %p", errors = "coerce")
    reference_date = pd.Timestamp("2026-06-01")
    df_clean["age"] = ((reference_date - df_clean["customer_birthdate"]).dt.days / 365.25)
    df_clean.loc[(df_clean["age"] < 16) | (df_clean["age"] > 100), "age"] = np.nan

    df_clean.loc[df_clean["year_first_transaction"] > 2026, "year_first_transaction"] = np.nan
    df_clean["tenure"] = 2026 - df_clean["year_first_transaction"]

    df_clean["has_loyalty_card"] = df_clean["loyalty_card_number"].notna().astype(int) # we assume the null values are the lack of a loyalty card

    df_clean["typical_hour_sin"] = np.sin(2 * np.pi * df_clean["typical_hour"] / 24)
    df_clean["typical_hour_cos"] = np.cos(2 * np.pi * df_clean["typical_hour"] / 24)

    df_clean.loc[df_clean["percentage_of_products_bought_promotion"] < 0, "percentage_of_products_bought_promotion"] = np.nan

    df_clean = df_clean.drop(columns = ["customer_name", "degree", "customer_gender", "customer_birthdate", "year_first_transaction", "loyalty_card_number",
                                      "typical_hour"])
    
    return df_clean
    

def fill_missing_values(df_clean):
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df_clean.select_dtypes(include = ["number"]).columns

    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    return df_clean


def plot_correlation_heatmap(df):
    plt.figure(figsize = (16, 12))
    sns.heatmap(df.select_dtypes(include = "number").corr(), cmap = sns.diverging_palette(220, 20, as_cmap = True), center = 0, annot = False,
                linewidths=0.5)
    plt.title("Correlation Matrix", fontsize = 14)
    plt.tight_layout()
    plt.show()