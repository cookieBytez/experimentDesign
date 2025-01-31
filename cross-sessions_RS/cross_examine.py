import cross_sessions_encode,cross_sessions_encode_evaluation
import cross_sessions_auto,cross_sessions_auto_evaluation
import cross_sessions_concat,cross_sessions_concat_evaluation
import GRU4REC_concat,GRU4REC_concat_evaluation
import GRU4REC,GRU4REC_evaluation
from datetime import datetime


def save_runtime():
    with open('evaluations/times_log.txt', 'a') as f:
    for time_entry in times:
        f.write(time_entry + "\n")

seeds = [21]#21,42,63,84]
rates = [0.1]#,0.2,0.3,0.4,0.5]
times = []

#cross_sessions_encode
# for i in seeds:
#     for j in rates:
#         start_time = datetime.now()
#         times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: cs_encode")
#         cross_sessions_encode.train_model(i,j)
#         cross_sessions_encode_evaluation.eval_model(i,j)
#         end_time = datetime.now()
#         times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: cs_encode")
#         times.append(f"Total time:{(end_time-start_time).total_seconds()}")

#GRU4REC
for i in seeds:
    for j in rates:
        start_time = datetime.now()
        times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: cs_encode")
        GRU4REC.train_model(i,j)
        GRU4REC_evaluation.eval_model(i,j)
        end_time = datetime.now()
        times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: cs_encode")
        times.append(f"Total time:{(end_time-start_time).total_seconds()}")
