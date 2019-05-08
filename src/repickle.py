#! /usr/bin/env python3s

import pickle

f2 = open('/home/mers/catkin_ws/src/terrain_classification/src/clf.pkl', 'rb')
clf = pickle.load(f2)

pickle.dump(clf, '/home/mers/catkin_ws/src/terrain_classification/src/clf0.pkl', protocol=0)