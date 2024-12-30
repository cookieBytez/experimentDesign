import numpy as np
import pandas as pd
import sys

sys.path.append('../extended')
import pre_processing_functions

''' Sessions. '''
data_sessions = pd.read_csv('../Data Sets/sessions_train.csv')

data_sessions = pre_processing_functions.one_hot_encode_actions_fit_transform(data_sessions)[0]

# Add column with start time for each session.
data_sessions['session_start'] = data_sessions.groupby(['event_id', 'session_id']).action_time.transform("min")
data_sessions = data_sessions.drop(['session_id'], axis=1)

group_columns = ['event_id']
sort_columns = ['session_start', 'action_time']
n_steps = 222
train_x, valid_x = pre_processing_functions.end_padding_and_split_sessions(data_sessions, group_columns, sort_columns, n_steps)

print(train_x)
print('hi')
print(valid_x)

train_x = train_x[:,:-1,:]
valid_x = valid_x[:,:-1,:]


object_columns = [col for col in data_sessions if col.startswith('action_object')]
data_sessions = data_sessions[['event_id', 'valid', 'session_start', 'action_time', *object_columns]]

group_columns = ['event_id']
sort_columns = ['session_start', 'action_time']
n_steps = 222
train_y, valid_y = pre_processing_functions.end_padding_and_split_sessions(data_sessions, group_columns, sort_columns, n_steps)

train_y = train_y[:,1:,:]
valid_y = valid_y[:,1:,:]