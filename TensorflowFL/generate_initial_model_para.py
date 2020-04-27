import numpy as np
import os


w = np.multiply(np.random.randn(784,10), 0.01)
b = np.add(np.multiply(np.random.rand(10), 0.01), -1)

f = open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"),"w")
f.close()

f = open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"),"a")

s = ""
for i in w:
    for j in i:
        s += str(j)+"\t"
s=s[0:-1]

print(s,file=f)

s = ""
for i in b:
    s+=str(i)+"\t"
s = s[0:-1]
print(s, file=f)

f.close()