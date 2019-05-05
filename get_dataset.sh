mkdir data
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -P data/
mkdir data/imgs
mkdir data/depths
python3 save_images.py
mkdir data/imgs_test data/imgs_val data/imgs_train data/depths_train data/depths_test data/depths_val
mv `ls data/imgs | head -1000` data/imgs_train
mv `ls data/imgs | head -200` data/imgs_val
mv `ls data/imgs` data/imgs_test
mv `ls data/depths | head -1000` data/depths_train
mv `ls data/depths | head -200` data/depths_val
mv `ls data/depths` data/depths_test


