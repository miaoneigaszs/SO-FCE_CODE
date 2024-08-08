import pandas as pd
import numpy as np

# Read data
y_train = pd.read_csv("data/y_train.txt")
df = pd.read_csv("data/X_train.txt", delimiter='\t')
df['label'] = y_train['label']

# List of error rates to generate
error_rates = [0.05, 0.10, 0.15, 0.20, 0.25]

for percentage_to_change in error_rates:
    df_copy = df.copy()
    df_copy['sign'] = 0

    # Find all indices where 'label' column is 1
    indices_to_change_1 = df_copy.index[df_copy['label'] == 1]
    # Randomly select indices to change
    indices_to_change_1 = np.random.choice(indices_to_change_1,
                                           size=int(percentage_to_change * len(indices_to_change_1)), replace=False)
    # Change the 'label' value to 0 for the selected indices
    df_copy.loc[indices_to_change_1, 'label'] = 0
    df_copy.loc[indices_to_change_1, 'sign'] = 1

    # Find all indices where 'label' column is 0 and 'sign' is 0
    indices_to_change_0 = df_copy.index[(df_copy['label'] == 0) & (df_copy['sign'] == 0)]
    # Randomly select indices to change
    indices_to_change_0 = np.random.choice(indices_to_change_0,
                                           size=int(percentage_to_change * len(indices_to_change_0)), replace=False)
    # Change the 'label' value to 1 for the selected indices
    df_copy.loc[indices_to_change_0, 'label'] = 1

    y_train_new = df_copy['label']
    df_copy.drop(['label', 'sign'], axis=1, inplace=True)

    # Save the generated files
    y_train_new.to_csv(f"data/y_train_{percentage_to_change}_wrong.txt", sep='\t', index=False)
    df_copy.to_csv(f"data/X_train_{percentage_to_change}_wrong.txt", sep='\t', index=False)
