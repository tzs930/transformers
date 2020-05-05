import numpy as np

loaded_arr = np.load('zork1_sapairs_maxdepth7.npy')

trainfile = open("zork1_sapairs_maxdepth7_train.txt","w")
evalfile = open("zork1_sapairs_maxdepth7_eval.txt","w")

len_total = len(loaded_arr)
idxs = np.arange(len_total)
trainidxs = np.random.choice(idxs, size=int(len_total*0.8),replace=False)
evalidxs = np.delete(idxs, trainidxs)
# testidxs = np.remove(idxs, trainidxs)
# import IPython; IPython.embed()

for elem in loaded_arr[trainidxs]:
    str1 = elem[0].replace('\n', ' ')
    str2 = "| [RESPONSE] " + elem[1] + '\n'
    trainfile.writelines(str1+str2)
    
for elem in loaded_arr[evalidxs]:
    str1 = elem[0].replace('\n', ' ')
    str2 = "| [RESPONSE] " + elem[1] + '\n'
    evalfile.writelines(str1+str2)

trainfile.close()
evalfile.close()