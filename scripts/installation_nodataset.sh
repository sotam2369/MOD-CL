cd /usr/src/MOD-CL/
unzip MaxHS.zip

cd /usr/src/road-dataset/road/
bash get_dataset.sh

cd ../
python extract_videos2jpgs.py /usr/src/road-dataset/road/

echo "" >> .gitignore
echo "road/rgb-images/" >> .gitignore
echo "road/videos/" >> .gitignore
echo "road_test/" >> .gitignore
echo "yolo_road-r_task1/" >> .gitignore
echo "yolo_road-r_task2/" >> .gitignore

mkdir road_test
cd road_test
mkdir videos
cd videos

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dTdvipm3Y9xEISvlqkzWfQisUzMGvC-V' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dTdvipm3Y9xEISvlqkzWfQisUzMGvC-V" -O 2014-06-26-09-31-18_stereo_centre_02.mp4 && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10eq0zDHInLCJS_sFfT2FApEeC86kEZ3K' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10eq0zDHInLCJS_sFfT2FApEeC86kEZ3K" -O 2014-12-10-18-10-50_stereo_centre_02.mp4 && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1D7a_T0K5Xko-eZOVRJvIAxi2FpENz7_C' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1D7a_T0K5Xko-eZOVRJvIAxi2FpENz7_C" -O 2015-02-03-08-45-10_stereo_centre_04.mp4 && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fYiOdAND2xyML9fEgMTdWnO1PQf8a8GN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fYiOdAND2xyML9fEgMTdWnO1PQf8a8GN" -O 2015-02-06-13-57-16_stereo_centre_01.mp4 && rm -rf /tmp/cookies.txt

cd ../../
python extract_videos2jpgs.py /usr/src/road-dataset/road_test/

cd /usr/src/