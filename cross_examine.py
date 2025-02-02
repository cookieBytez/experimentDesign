import models_code.cross_sessions_encode,models_code.cross_sessions_encode_evaluation
import models_code.cross_sessions_auto,models_code.cross_sessions_auto_evaluation
import models_code.cross_sessions_concat,models_code.cross_sessions_concat_evaluation
import models_code.GRU4REC_concat,models_code.GRU4REC_concat_evaluation
import models_code.GRU4REC,models_code.GRU4REC_evaluation
import visualizations.visualisations_LR
from datetime import datetime

def save_runtime(times,filename):
    with open(f'../runtime/times_log_{filename}.txt', 'a') as f:
        for time_entry in times:
            f.write(time_entry + "\n")


seeds = [42] #add more values to check models with different seed
rates = [0.1,0.2,0.3,0.4,0.5] #add more values to check with different learning rates


#popular_model
import models_code.popular_evaluation

#random_model
import models_code.random_evaluation

#GRU4REC model
for i in seeds:
    for j in rates:
        times = []
        print("GRU4REC")
        start_time = datetime.now()
        times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: GRU4REC")
        models_code.GRU4REC.train_model(i,j)
        models_code.GRU4REC_evaluation.eval_model(i,j)
        end_time = datetime.now()
        times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: GRU4REC")
        times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
        save_runtime(times,'GRU4REC')
    visualizations.visualisations_LR.visualise('GRU4REC',i)

#GRU4REC_concat model
for i in seeds:
    for j in rates:
        times = []
        print("GRU4REC_concat")
        start_time = datetime.now()
        times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: GRU4REC_concat")
        models_code.GRU4REC_concat.train_model(i,j)
        models_code.GRU4REC_concat_evaluation.eval_model(i,j)
        end_time = datetime.now()
        times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: GRU4REC_concat")
        times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
        save_runtime(times,'GRU4REC_concat')
    visualizations.visualisations_LR.visualise('GRu4REC_concat',i)

#cross_sessions_auto
for i in seeds:
    for j in rates:
        times = []
        print("cross_sessions_auto")
        start_time = datetime.now()
        times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: cs_auto")
        models_code.cross_sessions_auto.train_model(i,j)
        models_code.cross_sessions_auto_evaluation.eval_model(i,j)
        end_time = datetime.now()
        times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: cs_auto")
        times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
        save_runtime(times,'cs_auto')
    visualizations.visualisations_LR.visualise('auto',i)

#cross_sessions_encode
for i in seeds:
    for j in rates:
        times = []
        print("cross_sessions_encode")
        start_time = datetime.now()
        times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: cs_encode")
        models_code.cross_sessions_encode.train_model(i,j)
        models_code.cross_sessions_encode_evaluation.eval_model(i,j)
        end_time = datetime.now()
        times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: cs_encode")
        times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
        save_runtime(times,'cs_encode')
    visualizations.visualisations_LR.visualise('encode',i)

#cross_sessions_concat
for i in seeds:
    for j in rates:
        times = []
        print("cross_sessions_concat")
        start_time = datetime.now()
        times.append(f"Start: {start_time} - Seed: {i}, Rate: {j}, Model: cs_concat")
        models_code.cross_sessions_concat.train_model(i,j)
        models_code.cross_sessions_concat_evaluation.eval_model(i,j)
        end_time = datetime.now()
        times.append(f"End: {end_time} - Seed: {i}, Rate: {j}, Model: cs_concat")
        times.append(f"Total time:{(end_time-start_time).total_seconds()} secs")
        save_runtime(times,'cs_concat')
    visualizations.visualisations_LR.visualise('concat',i)
