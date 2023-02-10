catkin clean
sudo rm -rf .catkin_tools
catkin init
mv ./src/cv_bridge ./cv_bridge
catkin build	# build all other packages except cv_bridge using default python2

catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 \
-DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so

mv ./cv_bridge ./src/cv_bridge
catkin build cv_bridge	# build cv_bridge using python3
source ~/.bashrc
