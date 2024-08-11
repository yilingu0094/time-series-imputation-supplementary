import experiment_synthetic as exp
import time

time_li = []

#exp.general_experiment('rho',2)

for i in ['rho','delta']:
    for j in range(0,10):
        start_time = time.time()
        exp.general_experiment(i,j)
        #exp.specific_experiment(i,j)
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_li.append(elapsed_time)
time_li.append(sum(time_li))

# Open a file in write mode
with open('experiment_time.txt', 'w') as file1:
# Iterate over each item in the list
    for item in time_li:
# Write the item to the file followed by a newline character
        file1.write(f"\n{item}")     
