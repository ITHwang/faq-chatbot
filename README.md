# FAQ Chatbot

## 1. Setting Env

### 1.1. Environments

- Python 3.10.12
- RTX A6000-48G
- CUDA 12.0

### 1.2. Install libraries

```
pip install -r requirements.txst
```

## 2. Data ETL

```sh
python data/etl.py --pkl_path <pkl file path>
```

- After the cmd, json file and db file are saved on `./data`

## 3. Run

```
python run.py
```
