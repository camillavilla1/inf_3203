import matplotlib.pyplot as plt
import numpy as np
filename = "result"

text_file =open(filename, "r")
lines = text_file.readlines()

w1 = np.array([])
w2 = np.array([])
w3 = np.array([])
w4 = np.array([])
w5 = np.array([])
w6 = np.array([])
w7 = np.array([])
w8 = np.array([])
w9 = np.array([])
w10 = np.array([])
    
l1 = [elem.strip().split(',') for elem in lines]
#print l1
for i in l1:
    float(i[0])
    float(i[1])

for some in l1:
    #print some
    if (float(some[0]) == 1):
        w1 = np.append(w1, some[1])
    elif(float(some[0]) == 2):
        w2 = np.append(w2, some[1])
    elif(float(some[0]) == 3):
        w3 = np.append(w3, some[1])
    elif(float(some[0]) == 4):
        w4 = np.append(w4, some[1])
    elif(float(some[0]) == 5):
        w5 = np.append(w5, some[1])
    elif(float(some[0]) == 6):
        w6 = np.append(w6, some[1])
    elif(float(some[0]) == 7):
        w7 = np.append(w7, some[1])
    elif(float(some[0]) == 8):
        w8 = np.append(w8, some[1])
    elif(float(some[0]) == 9):
        w9 = np.append(w9, some[1])
    elif(float(some[0]) == 10):
        w10 = np.append(w10, some[1])
    else:
        pass
    


#MEAN
ww1 = np.array(w1).astype(np.float)
ww2 = np.array(w2).astype(np.float)
ww3 = np.array(w3).astype(np.float)
ww4 = np.array(w4).astype(np.float)
ww5 = np.array(w5).astype(np.float)
ww6 = np.array(w6).astype(np.float)
ww7 = np.array(w7).astype(np.float)
ww8 = np.array(w8).astype(np.float)
ww9 = np.array(w9).astype(np.float)
ww10 = np.array(w10).astype(np.float)

w1m = np.mean(ww1)
w2m = np.mean(ww2)
w3m = np.mean(ww3)
w4m = np.mean(ww4)
w5m = np.mean(ww5)
w6m = np.mean(ww6)
w7m = np.mean(ww7)
w8m = np.mean(ww8)
w9m = np.mean(ww9)
w10m = np.mean(ww10)

#print w1m, w2m,w3m,w4m,w5m, w6m, w7m,w8m,w9m,w10m

#STANDARD DEVIATION
w1s = np.std(ww1)
w2s = np.std(ww2)
w3s = np.std(ww3)
w4s = np.std(ww4)
w5s = np.std(ww5)
w6s = np.std(ww6)
w7s = np.std(ww7)
w8s = np.std(ww8)
w9s = np.std(ww9)
w10s = np.std(ww10)

#print('\n')
#print w1s, w2s,w3s,w4s,w5s, w6s, w7s, w8s, w9s,w10s

#y= [19,20, 25, 15, 10]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mn = [w1m, w2m, w3m, w4m, w5m, w6m, w7m, w5m, w9m, w10m]
ws =[w1s, w2s, w3s, w4s, w5s, w6s, w7s, w8s, w9s, w10s]

plt.plot(x, mn)
plt.xlim(0.5, 10.5)
#plt.ylim()
#fig1 = plt.figure(1)
plt.xlabel("Number of workers")
plt.ylabel("Time executing in sec")
plt.errorbar(x, mn, ws, fmt='o') #x, mn, ws
#plt.title("Working on it")
plt.savefig('result', format='pdf')
plt.savefig('result', format='png')
plt.show()
        
text_file.close()
