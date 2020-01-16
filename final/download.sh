rm -rf checkpoints
mkdir checkpoints
wget https://www.dropbox.com/s/6rgadk5c3v3qzzl/ml2019fall_models_part_1.zip
unzip ml2019fall_models_part_1.zip
mv ml2019fall_models_part_1/* checkpoints
rm -rf ml2019fall_models_part_1*
wget https://www.dropbox.com/s/xt4vz83d7xpsbs7/ml2019fall_models_part_2.zip
unzip ml2019fall_models_part_2.zip
mv ml2019fall_models_part_2/* checkpoints
rm -rf ml2019fall_models_part_2*
