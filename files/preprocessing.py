def preprocess(df):

    if 'LoanID' in df.columns:
        df = df.drop('LoanID', axis=1)

    # Fixed: ffill() replaces deprecated fillna(method='ffill')
    df = df.ffill()

    # Fixed: .apply + lambda replaces deprecated .applymap
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    mappings = {
        'Education':      {"High School": 0, "Bachelor's": 1, "Master's": 2},
        'EmploymentType': {"Unemployed": 0, "Part-time": 1, "Full-time": 2},
        'MaritalStatus':  {"Single": 0, "Married": 1, "Divorced": 2},
        'HasMortgage':    {"No": 0, "Yes": 1},
        'HasDependents':  {"No": 0, "Yes": 1},
        'LoanPurpose':    {"Auto": 0, "Business": 1, "Home": 2, "Other": 3},
        'HasCoSigner':    {"No": 0, "Yes": 1},
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = df.fillna(0)
    return df
