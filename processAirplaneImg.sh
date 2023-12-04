. .venv/bin/activate
pip install -r requirements.txt
# --imageDir="/Volumes/Untitled/DCIM/100MEDIA"
python3 mobileNetImageProcessing.py --imageDir="/media/moose/disk/DCIM/100MEDIA" \
    --baseDir=/home/moose/Pictures \
    --airplaneCanidateDir="airPlaneCanidates" \
    --carDir="carCanidates"

