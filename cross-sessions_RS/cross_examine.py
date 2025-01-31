import cross_sessions_encode,cross_sessions_encode_evaluation
import cross_sessions_auto,cross_sessions_auto_evaluation
import cross_sessions_concat,cross_sessions_concat_evaluation
import GRU4REC_concat,GRU4REC_concat_evaluation
import GRU4REC,GRU4REC_evaluation
from datetime import datetime
import os

def save_runtime(times,filename):
    with open(f'evaluations/times_log_{filename}.txt', 'a') as f:
        for time_entry in times:
            f.write(time_entry + "\n")
    


seeds = [42]#       [21,42,63,84]// set seed sccordingly
rates = [0.1,0.2,0.3,0.4,0.5]#      [0.1,0.2,0.3,0.4,0.5]// set learning rates sccordingly

#Uncomment the models which needs to be run


#popular_model
#import popular_evaluation

#random_model
#import random_evaluation

#GRU4REC
# for i in seeds:
#     for j in rates:
#         times = []
#         print("GRU4REC")
#         start_time = datetime.now()
#         times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: GRU4REC")
#         GRU4REC.train_model(i,j)
#         GRU4REC_evaluation.eval_model(i,j)
#         end_time = datetime.now()
#         times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: GRU4REC")
#         times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
#         save_runtime(times,'GRU4REC')

#GRU4REC_concat
# for i in seeds:
#     for j in rates:
#         times = []
#         print("GRU4REC_concat")
#         start_time = datetime.now()
#         times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: GRU4REC_concat")
#         GRU4REC_concat.train_model(i,j)
#         GRU4REC_concat_evaluation.eval_model(i,j)
#         end_time = datetime.now()
#         times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: GRU4REC_concat")
#         times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
#         save_runtime(times,'GRU4REC')

#cross_sessions_encode
for i in seeds:
    for j in rates:
        times = []
        print("cross_sessions_encode")
        start_time = datetime.now()
        times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: cs_encode")
        cross_sessions_encode.train_model(i,j)
        cross_sessions_encode_evaluation.eval_model(i,j)
        end_time = datetime.now()
        times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: cs_encode")
        times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
        save_runtime(times,'cs_encode')

#cross_sessions_auto
# for i in seeds:
#     for j in rates:
#         times = []
#         print("cross_sessions_auto")
#         start_time = datetime.now()
#         times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: cs_auto")
#         cross_sessions_auto.train_model(i,j)
#         cross_sessions_auto_evaluation.eval_model(i,j)
#         end_time = datetime.now()
#         times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: cs_auto")
#         times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
#         save_runtime(times,'cs_auto')

#cross_sessions_concat
# for i in seeds:
#     for j in rates:
#         times = []
#         print("cross_sessions_concat")
#         start_time = datetime.now()
#         times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: cs_concat")
#         cross_sessions_concat.train_model(i,j)
#         cross_sessions_concat_evaluation.eval_model(i,j)
#         end_time = datetime.now()
#         times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: cs_concat")
#         times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
#         save_runtime(times,'cs_concat')



