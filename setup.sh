mkdir -p data_sets/flowers
mkdir -p models/flower_model
curl https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz --output data_sets/flowers/flower_photos.tgz
curl https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg --output 592px-Red_sunflower.jpg
cd data_sets/flowers
tar -xzf flower_photos.tgz
rm flower_photos.tgz
cd ../../
mkdir -p models/flower_model/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Installing numpy on an M1/M2 Mac??


pip install -r requirements.txt
