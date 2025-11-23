# VLM_BCR_Prediction
Exploring Medical Multimodal Models for Predicting Breast Cancer Risk 

## 1) Pre-installations needed in your system
Install the following softwares and verify if they are properly installed in your system.

- Install [Python 3.10.12](https://www.python.org/downloads/release/python-3117/) 

- Install [Git](https://git-scm.com/downloads) 

## 2) Clone the git repository
Run the following commands in your command prompt
```
git clone https://github.com/RaiyanJahangir/VLM_BCR_Prediction.git
```

## 3) Go to the root directory of the project
```
cd VLM_BCR_Prediction
```

## 4) Create a directory to store the data 
```
mkdir embed
```

## 5) Enter the directory 
```
cd embed
```

## 6) Download the dataset

- Download the EMBED dataset from [Amazon AWS](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/).
- Store the dataset in the following way inside the project directory:

```
VLM_BCR_Prediction/
├─ embed/
│  ├─ png_images/..
│  ├─ csv files 
│  
├─ Other Python Codes
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## 7) Create a virtual environment (myenv)
You may rename the virtual environment as anything you like.

For Windows
```
py -3.10 -m venv myenv  
```

For Linux
```
python3.10 -m venv myenv 
```

## 8) Activate the virtual environment 

For Windows
```
myenv/scripts/activate
```

For Linux
```
source myenv/bin/activate
```

## 9) Install all the necessary packages and libraries
```
pip install -r requirements.txt
```

## 9) Prepare the JSON Report Format
For Windows
```
python make_per_patient_json.py
```

For Linux
```
python3 make_per_patient_json.py
```

## Deactivate virtual environment and wrap up
```
deactivate
```
