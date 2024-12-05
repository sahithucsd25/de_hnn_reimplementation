# DE-HNN Re-implementation
```
de_hnn_reimplementation/
├── data/ - Store processed_datasets here
├── plots/ - Contains loss plots
├── results/ - Contains loss data, evaluation metrics, and best model params
├── src/ - Contains the main codebase
│   ├── dehnn_layers.py
│   ├── model.py
│   ├── pyg_dataset.py
│   ├── train.py
│   ├── utils.py
├── environment.yml
├── README.md
└── run.py - Build script
```

### Retrieving the data

(1) To download the data, simply enter the following url and download the processed_datasets.zip file:

https://drive.google.com/file/d/1VactdnhGDFuOjdkJxp1g1ra-2n1IaFuV/view?usp=drive_link

(2) Unzip processed_datasets.zip and place processed_datasets inside __data/__

### Running the project

* This project assumes processed_dataset is unzipped and stored in __data/__
* To setup the environment using conda: `conda env create -f environment.yml`
This will create an environment called de_hnn.

### Training/Evaluating the model using `run.py`

* `run.py` takes arguments in the form of `<test> <n_epochs>`
  - `<test>` is a boolean
  - `<n_epochs>` is an integer
* To test the model, from project root dir, run `python run.py true`
  - This evaluates model using the best model params stored in __results/__
* To train a new model, from project root dir, run `python run.py false <n_epochs>`, where `<n_epochs>` is an integer
  - This will train a new model and save its results in __results/__
  - This will save new loss plots given the training and validation losses