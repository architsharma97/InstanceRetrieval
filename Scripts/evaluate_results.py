import sys
results_file = sys.argv[1]
k = int(sys.argv[2])

with open(results_file,'r') as f:
	predictions = f.read().split('\n')[:-1]
prediction_list = []

for prediction in predictions:
	prediction_list.append(prediction.split(','))

map_final = 0
for i in range(len(prediction_list)):
	correct = 0
	recall = 0
	map_val = 0
	for j in range(k):
		if (i)/9 == (int(prediction_list[i][j]))/63:
			correct+=1
			precision = float(correct)/(j+1)
			recall += 1.0/k
			map_val+=precision*recall
			print(correct)


	map_final += map_val
map_final = map_final/k
map_final = map_final/(84*9)
			
print(map_final)
