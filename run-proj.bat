@echo off

echo "Preparing to run project..."
echo "Setting up environment..."
pipenv sync
pipenv shell

echo "Preprocessing the Dataset..."
python preprocess.py --input-dir Data --output-dir output/intermediate --crop-dim 180

echo "Training Classifier..."
echo "This is a long running process, will take some time..."
python --input-dir output/intermediate --model-path etc/20170511-185253/20170511-185253.pb --classifier-path output/classifier.pkl --num-threads 16 --num-epochs 25 --min-num-images-per-class 10 --is-train