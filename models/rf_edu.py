



input_train_file="complete_output.npy"
target_train_file="s2.npy"
ioptpredict=0
max_depth=3

if ioptpredict==1:
	input_predict_file="slp_predict.npy"
	output_predict_file="pr_predict.npy"

import numpy as np
from sklearn.ensemble import RandomForestClassifier


x=np.load(input_train_file)
y=np.load(target_train_file)
(t,ndim)=(x.shape)
if ioptpredict==1:
	z=np.load(input_predict_file)


iy=y.astype(int)

nclass=len(set(y))
print(nclass)

clf = RandomForestClassifier(n_estimators=100,class_weight="balanced",oob_score=True,max_depth=max_depth,random_state=0)
clf.fit(x, iy)
print(clf.feature_importances_)
np.save('importances',clf.feature_importances_)

if ioptpredict==1:
	myz=clf.predict(z)
	np.save(output_predict_file,myz)
