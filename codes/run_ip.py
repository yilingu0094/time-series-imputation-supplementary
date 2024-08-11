import experiment_ind_portfolio as exp
import time

time_li = []

for i in ['rho','delta']:
    if i == 'rho':
        for j in range(2,4):
            start_time = time.time()
            exp.general_experiment(i,j)#j is the number of the given delta
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_li.append(elapsed_time)
    else:
        for j in range(2,4):
            start_time = time.time()
            exp.general_experiment(i,j)#j is the number of the given rho
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_li.append(elapsed_time)

time_li.append(sum(time_li))

# Open a file in write mode
with open('experiment_time_ip.txt', 'w') as file1:
# Iterate over each item in the list
    for item in time_li:
# Write the item to the file followed by a newline character
        file1.write(f"\n{item}")       