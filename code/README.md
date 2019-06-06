## Usage Instructions
### 1. Process the EuroSense Dataset
```
python preprocess_ES.py parent_path
```
where the **required** arguments are:
- parent_path: Path to the eurosense.v1.0.high-precision **(.xml)** file


### 2. Train Word2Vec model
#### 2.1 Train Word2Vec model for a single experiment
```
python train.py embed_fname
```
where the **required** arguments are:
- embed_fname: Path where the embeddings **(.vec)** file will be saved

#### 2.2 Train Word2Vec model using Grid Search Algorithm
```
python grid_search.py test_f_path
```
where the **required** arguments are:
- test_f_path: Path to a Word Similarity dataset **(.tab)** file

### 3. Test the model on Word Similarity Dataset
```
python test.py -m Test test_f_path [embed_fname]
```
where:

The **required** arguments are:
- -m / --mode: Wether to "Test" embeddings or to "Plot" some of them
- "-t" / "--test_f_path": Path to a Word Similarity dataset **(.tab)** file

while the *optional* are:
- -e / --embed_path: embeddings **(.vec)** file to test; will load embeddings from */resources/embeddings.vec* if no argument is given

### 4. Plot the sense embeddings of a view vectors
```
python test.py -m Plot [-e ..\resources\embeddings.vec]
```
where:

The **required** arguments are:
- -m / --mode: Wether to "Test" embeddings or to "Plot" some of them

while the *optional* are:
- -e / --embed_path: embeddings **(.vec)** file to use to plot; will load embeddings from *../resources/embeddings.vec* if no argument is given
