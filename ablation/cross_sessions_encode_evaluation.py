import pandas as pd
import argparse
# workarounds
import sys
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.append('../cross-sessions_RS')
import pre_processing_functions
import numpy as np
from tensorflow.keras.models import load_model
import evaluation_functions

# Set up argument parser
parser = argparse.ArgumentParser(description='Train and evaluate a model using specified dataset.')
parser.add_argument('csv_filename', type=str, help='CSV file name for input data')
parser.add_argument('purchase_events_csv', type=str, help='CSV file name for purchase events data')
parser.add_argument('filter_csv', type=str, help='CSV file name for purchase events data')
parser.add_argument('name', type=str, help='CSV file name for purchase events data')

# Parse arguments
args = parser.parse_args()
csv_filename = args.csv_filename
purchase_events_csv = args.purchase_events_csv
filter_csv = args.filter_csv
name = args.name


''' Sessions. '''
data_sessions = pd.read_csv(csv_filename)
print(csv_filename)

data_sessions, dummy_names = pre_processing_functions.one_hot_encode_actions_transform(data_sessions)

# Max pooling operation   
agg_dict = dict.fromkeys(dummy_names, 'max')
agg_dict.update(dict.fromkeys(['action_time'], 'min'))   
data_sessions = data_sessions.groupby(['event_id', 'session_id'], as_index=False).agg(agg_dict)          
data_sessions = data_sessions.drop(['session_id'], axis=1)  

group_columns = ['event_id']
sort_columns = ['action_time']
n_steps = 7
test_x = pre_processing_functions.start_padding_and_split_sessions(data_sessions, group_columns, sort_columns, n_steps, split=False)


''' Purchase events. '''
data_events = pd.read_csv(purchase_events_csv)

data_events = data_events.drop(['user_id', 'event_time'], axis=1)
test_y = pre_processing_functions.binarize_and_split_purchases(data_events, split=False)[0]

''' Filter. '''
data_filter = pd.read_csv(filter_csv)

test_w = pre_processing_functions.binarize_and_split_purchases(data_filter, split=False)[0]


''' Predict. '''
model = load_model(f'{name}_encode_model.h5')
pred = model.predict(test_x)
pred = pred*test_w


''' Evaluation. '''
k = 3

hit =  evaluation_functions.hit(pred, test_y, k)

precision = evaluation_functions.precision(pred, test_y, k)

recall = evaluation_functions.recall(pred, test_y, k)

rr = evaluation_functions.reciprocal_rank(pred, test_y, k)

ap = evaluation_functions.average_precision(pred, test_y, k)

print([np.mean(hit), np.mean(precision), np.mean(recall), np.mean(rr), np.mean(ap)])


# Statistical significans
statistical_significans = pd.DataFrame({'hit' : hit, 'precision' : precision, 'recall' : recall, 'RR' : rr, 'AP' : ap})
statistical_significans.to_csv(f'{name}_statistical_significans_encode.csv', index=False)

# Varying thresholds
hr = []
precision = []
recall = []
mrr = []
mean_average_precision = []
for k in range(1,6):
    hr.append(np.mean(evaluation_functions.hit(pred, test_y, k)))
    precision.append(np.mean(evaluation_functions.precision(pred, test_y, k)))
    recall.append(np.mean(evaluation_functions.recall(pred, test_y, k)))
    mrr.append(np.mean(evaluation_functions.reciprocal_rank(pred, test_y, k)))
    mean_average_precision.append(np.mean(evaluation_functions.average_precision(pred, test_y, k)))

varying_thresholds = pd.DataFrame({'HR' : hr, 'precision' : precision, 'recall' : recall, 'MRR' : mrr, 'MAP' : mean_average_precision})
varying_thresholds.to_csv(f'{name}_varying_thresholds_encode.csv', index=False)
