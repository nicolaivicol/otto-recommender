brew install python3.11
python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv .venv-otto-recommender --python $(which python3.11)
source .venv-otto-recommender/bin/activate
pip install -r requirements.txt
