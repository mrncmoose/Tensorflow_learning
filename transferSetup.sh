image_dir="data_sets/cats_and_dogs_images"
model_dir="models/cats_and_dogs_model"
img_zip_file="cats_and_dogs_filtered.zip"
model_net_model='mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5'

mkdir -p $image_dir
mkdir -p $model_dir

#curl https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip --output "$image_dir/$img_zip_file"
curl https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5 --output "$model_dir/$model_net_model"
#cd $image_dir
#unzip $img_zip_file
#rm $img_zip_file
#cd ../../


