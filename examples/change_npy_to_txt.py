import numpy as np

loaded_arr = np.load('merged_pairs.npy')

trainfile = open("merged_pairs_train.txt","w")
evalfile = open("merged_pairs_eval.txt","w")

len_total = len(loaded_arr)
subsample_ratio = 0.4
total_idxs = np.arange(len_total)
subsample_idxs = np.random.choice(total_idxs, size=int(len_total * subsample_ratio), replace=False)
subsample_arr = loaded_arr[subsample_idxs]

len_subsample = len(subsample_arr)
idxs = np.arange(len_subsample)
trainidxs = np.random.choice(idxs, size=int(len_subsample*0.8),replace=False)
evalidxs = np.delete(idxs, trainidxs)
# testidxs = np.remove(idxs, trainidxs)
# import IPython; IPython.embed()

for elem in subsample_arr[trainidxs]:
    str1 = elem[0].replace('\n', ' ')
    str2 = "| [RESPONSE] " + elem[1] + '\n'
    trainfile.writelines(str1+str2)
    
for elem in subsample_arr[evalidxs]:
    str1 = elem[0].replace('\n', ' ')
    str2 = "| [RESPONSE] " + elem[1] + '\n'
    evalfile.writelines(str1+str2)

trainfile.close()
evalfile.close()
