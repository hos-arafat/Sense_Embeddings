## Submission skeleton
```
- code               # put your training and evaluation code here
- resources
  |__ embeddings.vec # substitute this file with your embeddings
- README.md          # this file
- Homework_2_nlp.pdf # the slides presenting this homework
- report.pdf         # your report
```
## Usage Instructions
### Process the EuroSense Dataset
``` python preprocess_ES.py [parent_path]
```
where:
```parent_path: Path to the eurosense.v1.0.high-precision.xml file
```

### Train Word2Vec model
``` python train.py [embed_fname]
```
where:
```
embed_fname: Path where the embeddings (.vec) file will be saved
 ```
### Test the model on Word Similarity Dataset
``` python test.py Test [test_f_path] [embed_fname]
```
where:
```
embed_fname: Path to a Word Similarity dataset (.tab) file
embed_fname: Path to embeddings (.vec) file will to test
 ```
