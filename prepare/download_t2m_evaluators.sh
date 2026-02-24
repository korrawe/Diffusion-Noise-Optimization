echo -e "Downloading T2M evaluators"
gdown --fuzzy https://drive.google.com/file/d/1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb/view
gdown --fuzzy https://drive.google.com/file/d/12liZW5iyvoybXD8eOw4VanTgsMtynCuU/view
rm -rf t2m
rm -rf kit

unzip t2m.zip
unzip kit.zip
echo -e "Cleaning\n"
rm t2m.zip
rm kit.zip

echo -e "Downloading done!"