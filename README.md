# VLM_BCR_Prediction
Exploring Medical Multimodal Models for Predicting Breast Cancer Risk 

## 1) Pre-installations needed in your system
Install the following softwares and verify if they are properly installed in your system.

- Install [Python 3.10.12](https://www.python.org/downloads/release/python-3117/) 

- Install [Git](https://git-scm.com/downloads) 

- Install [Ollama](https://ollama.com/)

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
## 7) Come back to the main directory
```
cd ..
```

## 8) Create a virtual environment (myenv)
You may rename the virtual environment {myenv} to your choice.

For Windows
```
py -3.10 -m venv myenv  
```

For Linux
```
python3.10 -m venv myenv 
```

## 9) Activate the virtual environment 

For Windows
```
myenv/scripts/activate
```

For Linux
```
source myenv/bin/activate
```

## 10) Install all the necessary packages and libraries
```
pip install -r requirements.txt
```

## 11) Prepare the JSON Report Format
For Windows
```
python make_per_patient_json.py
```

For Linux
```
python3 make_per_patient_json.py
```

## 12) Run the Models on Zero Shot Prompt
- We have used 3 models for our work. You can download any models from Ollama.
- model_name = {qwen2.5vl, blaifa/InternVL3, anas/video-llava:test}
For Linux
```
python3 zeroshot_risk.py \
  --model {model_name} \
  --report_dir {directory where the reports are saved} \
  --output_dir {directory where the generated reports will be saved} \
  --timeout 600 \
  --num -1
```
For Windows, write "python" instead of "python3"

## 13) Run the models on Few Shot Prompt
```
python3 few_risk.py \
  --model {model_name} \
  --report_dir {directory where the reports are saved} \
  --output_dir {directory where the generated reports will be saved} \
  --timeout 600 \
  --num -1
```

## 14) Run the models on CoT Prompt
```
python3 cot_risk.py \
  --model {model_name} \
  --report_dir {directory where the reports are saved} \
  --output_dir {directory where the generated reports will be saved} \
  --timeout 600 \
  --num -1
```

## 15) Run the models on ToT Prompt
```
python3 tot_risk.py \
  --model {model_name} \
  --report_dir {directory where the reports are saved} \
  --output_dir {directory where the generated reports will be saved} \
  --timeout 600 \
  --num -1
```

## 16) Evaluate and save the results
```
python3 evaluate_risk_metrics.py \
  --gt_dir {directory of ground truth reports} \
  --pred_dir {directory of generated reports} \
  --horizon_years 8 \
  --threshold 0.2 \
  --bootstrap 1000
```


## 17) Deactivate virtual environment and wrap up
```
deactivate
```
