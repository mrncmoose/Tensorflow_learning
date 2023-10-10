. .venv/bin/activate
pip install -r mac_requirements.txt
# --imageDir="/Volumes/Untitled/DCIM/100MEDIA"
python3 mobileNetImageProcessing.py --imageDir="/Users/moose/Pictures/gameCamImages" \
    --baseDir=/Users/moose/Pictures/SeptPlaneImages \
    --airplaneCanidateDir="airPlaneCanidates" \
    --carDir="carCanidates"

