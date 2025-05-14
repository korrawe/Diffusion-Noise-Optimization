mkdir -p body_models
cd body_models/

echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
rm -rf smpl

unzip smpl.zip
# Download additional NPZ converted body model
wget -O smpl/SMPL_NEUTRAL.npz https://polybox.ethz.ch/index.php/s/cjLQ8dRwTFoQZAG/download/SMPL_NEUTRAL.npz
echo -e "Cleaning\n"
rm smpl.zip

echo -e "Downloading done!"
