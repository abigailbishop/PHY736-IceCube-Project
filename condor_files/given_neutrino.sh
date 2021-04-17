#!/bin/bash
# project.sh
# $1 = Energy
# $2 = Outfolder
# $3 = Oufile

source /home/abishop/.bash_profile-condor
cvmfs icecube

echo "Running simulation"
python /home/abishop/736_project/PHY736-IceCube-Project/project.py -l 20 -21 20 -d 0 1 0 -e $1 -g 'IceCube' -o /data/user/abishop/736_project$2/$3 -b True -u abishop
