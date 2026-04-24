def preprocess(df):

    if 'LoanID' in df.columns:
        df = df.drop('LoanID', axis=1)

    # Fixed: ffill() replaces deprecated fillna(method='ffill')
    df = df.ffill()

    # Fixed: .apply + lambda replaces deprecated .applymap
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    mappings = {
        'Gender': {'Male': 0, 'Female': 1},
        'Married': {'No': 0, 'Yes': 1},
        'Education': {'Not Graduate': 0, 'Graduate': 1},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = df.fillna(0)
    return df
