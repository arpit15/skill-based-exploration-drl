#!/bin/bash
sudo apt-get install libnlopt-dev git python-catkin-pkg vim ros-kinetic-desktop-full swig ros-kinetic-kdl-parser

echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

pip install virtualenv ipdb

export ENVHOME=${HOME}/new_RL3
virtualenv -p python3.5 $ENVHOME

source ${ENVHOME}/bin/activate
pip install catkin_pkg
mkdir -p ${HOME}/my_ws/src
cd ${HOME}/my_ws/src

git clone https://arpit_agarwal_@bitbucket.org/arpit_agarwal_/trac_ik.git
cd ${HOME}/my_ws/src/trac_ik
git checkout python_wrapper_separated_pkg
cd ${HOME}/my_ws
catkin_make -DPYTHON_EXECUTABLE=${ENVHOME}/bin/python
ln -s ${HOME}/my_ws/devel/lib/python3/dist-packages/trac_ik_python ${ENVHOME}/lib/python3.5/site-packages

cd ${ENVHOME}
pip install mpi4py tensorflow-gpu==1.3.0 gym mujoco_py==0.5.7
git clone https://arpit_agarwal_@bitbucket.org/arpit_agarwal_/hindsight_experience_replay.git
cd ${ENVHOME}/hindsight_experience_replay
ln -s ${ENVHOME}/hindsight_experience_replay/HER ${ENVHOME}/lib/python3.5/site-packages
