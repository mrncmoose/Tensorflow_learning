. .venv/bin/activate
pip install -r mac_requirements.txt
python3 mobileNetImageProcessing.py --imageDir="/Volumes/Untitled/DCIM/100MEDIA" \
    --baseDir=/Users/moose/Pictures/AugustPlaneImages \
    --airplaneCanidateDir="airPlaneCanidates" \
    --carDir="carCanidates"

