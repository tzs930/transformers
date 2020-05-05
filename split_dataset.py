import numpy as np

A = np.load('merged_pairs.npy')
idxs = np.arange(len(A))
train_idxs = np.random.choice(idxs, int(len(A)*0.8), replace=False)
np.random.shuffle(train_idxs)
#import pdb; pdb.set_trace()
test_idxs = np.delete(idxs, train_idxs)
np.random.shuffle(test_idxs)

print('- num_train : %d, num_eval : %d'%(len(train_idxs), len(test_idxs)))
np.save('train_data.npy', A[train_idxs])
np.save('eval_data.npy', A[test_idxs])