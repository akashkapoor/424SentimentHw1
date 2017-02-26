import os
import sys

i = 1 if len(sys.argv) < 5 else int(sys.argv[4])
with open(sys.argv[2], 'a') as outfile:
	for filename in os.listdir(sys.argv[1]):
		with open(os.path.join(sys.argv[1], filename), 'r') as f:
			outfile.write("     " + str(i) + '\t' + f.readlines()[0] + '\t' + sys.argv[3] + os.linesep)
			i += 1
