# cloud-removal

##requirements

To install dependencies:
```bash
pip install -r requirements.txt
```

Datasets:
- Sen2_MTC: [CTGAN.zip](https://drive.google.com/file/d/1-hDX9ezWZI2OtiaGbE8RrKJkN1X-ZO1P/view?usp=share_link)

## train: 
```train
python run.py -p train -c config/ours_sigmoid.jaon
```
## test: 
```bash
python run.py -p test -c config/ours_sigmoid.json
```
## evaluation:
```bash
python ecaluation/eval.py -s [ground-truth image path] -d [predicted-sample image path]
