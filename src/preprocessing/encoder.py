import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Convert Label columns from text to numbers.
def Label_encoding(df: pd.DataFrame) -> pd.DataFrame :
    encoder = LabelEncoder()
    #Convert labels to numbers.
    df['Label'] = pd.Series(encoder.fit_transform(df['Label']))

    return df
