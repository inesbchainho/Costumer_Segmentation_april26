def preprocess_customer_info(df):
    print("----------------Handling missing values:-------------------")
    print(len(df[df.isnull().any(axis=1)]))
    # Deleting the one row with Credit Limit's null value
    df_dropped = df.dropna(subset = ['CREDIT_LIMIT'])
    # Filling in the rows with Minimum Payments' null values with the median, since its distribution is skewed:
    df_dropped['MINIMUM_PAYMENTS'] = df_dropped['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
