mkdir -p checkpoints/
cd checkpoints/
echo -e "The pretrained model motiongpt3.ckpt will stored in the 'checkpoints' folder\n"
# motiongpt3.ckpt
gdown --fuzzy "https://drive.google.com/file/d/1Wvx5PGJjVKPRvjcl8firChw1UVjUj36l/view?usp=drive_link" -O "checkpoints/motiongpt3.ckpt"


echo -e "Downloading done!"

