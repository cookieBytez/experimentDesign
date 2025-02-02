import pandas as pd
import os

def remove_category_rows(file_path, category, column_name):
    """
    Reads a CSV file, removes entire rows where the specified category is found
    in the provided column, and saves the modified DataFrame back to a new CSV.
    
    Parameters:
    - file_path: The path to the input CSV file.
    - category: The category to filter out (string).
    - column_name: The column name where the category should be found (string).
    
    Returns:
    - A set of event_ids that were removed.
    """
    df = pd.read_csv(file_path)

    # Find event_ids to remove
    event_ids_to_remove = set(df[df[column_name].astype(str).str.contains(category, case=False, na=False)]['event_id'])

    # Drop rows where event_id is in event_ids_to_remove
    df_filtered = df[~df['event_id'].isin(event_ids_to_remove)]

    base_name, ext = os.path.splitext(file_path)
    new_file_name = f"{base_name}_filtered_{category}{ext}"

    # Save the modified DataFrame to a new CSV file
    df_filtered.to_csv(new_file_name, index=False)
    print(f"Rows containing '{category}' in column '{column_name}' have been removed completely.")
    print(f"Modified file saved as: {new_file_name}")

    return event_ids_to_remove


def remove_events_from_purchase_events(file_path, category, event_ids_to_remove):
    """
    Reads the purchase events file, removes rows with event_ids that were removed in sessions,
    and saves the modified DataFrame back to a new CSV.
    
    Parameters:
    - file_path: The path to the purchase events CSV file.
    - event_ids_to_remove: A set of event_ids to remove.
    """
    df = pd.read_csv(file_path)

    # Drop rows where event_id is in event_ids_to_remove
    df_filtered = df[~df['event_id'].isin(event_ids_to_remove)]

    base_name, ext = os.path.splitext(file_path)
    new_file_name = f"{base_name}_filtered_{category}{ext}"

    # Save the modified DataFrame to a new CSV file
    df_filtered.to_csv(new_file_name, index=False)
    print(f"Rows with event_ids removed from sessions have been removed in purchase_events.")
    print(f"Modified purchase events file saved as: {new_file_name}")


# File paths
traintest = 'test'
sessions_file_path = f'../Data Sets/sessions_{traintest}.csv'
purchase_events_file_path = f'../Data Sets/purchase_events_{traintest}.csv'
filter_file_path = f'../Data Sets/filter_{traintest}.csv'

# List of categories and columns to remove rows from in sessions
categories_to_remove = [
    ('e_commerce', 'action_section'),
    ('claims_reporting', 'action_section'),
    ('information', 'action_section'),
    ('personal_account', 'action_section'),
    ('item', 'action_object'),
    ('service', 'action_object'),
    ('start', 'action_type'),
    ('act', 'action_type'),
    ('complete', 'action_type')
]

# Track event_ids to remove
all_event_ids_to_remove = set()

# Apply filtering for each category and column
for category, column_name in categories_to_remove:
    event_ids_removed = remove_category_rows(sessions_file_path, category, column_name)
    all_event_ids_to_remove.update(event_ids_removed)
    remove_events_from_purchase_events(purchase_events_file_path, category, event_ids_removed)
    remove_events_from_purchase_events(filter_file_path, category, event_ids_removed)

