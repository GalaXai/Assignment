#Assignment

### Assignment summary
1. Load the Covertype Data Set
  - https://archive.ics.uci.edu/ml/datasets/Covertype
2. Implement a very simple heuristic that will classify the data
  - It doesn't need to be accurate
3. Use Scikit-learn library to train two simple Machine Learning models
  - Choose models that will be useful as a baseline
4. Use TensorFlow library to train a neural network that will classify the data
  - Create a function that will find a good set of hyperparameters for the NN
  - Plot training curves for the best hyperparameters
5. Evaluate your neural network and other models
  - Choose appropriate plots and/or metrics to compare them
6. Create a very simple REST API that will serve your models
  - Allow users to choose a model (heuristic, two other baseline models, or neural network)
  - Take all necessary input features and return a prediction
  - Do not host it anywhere, the code is enough


### Usage

```bash
python app.py 
```

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model": "Neural Network", "features": [2699,347,3,0,0,2096,213,234,159,6853,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}' http://127.0.0.1:5000/predict
```
returns
```bash
{"prediction":1}
```
#### With Decode
```bash
curl -X POST -H "Content-Type: application/json" -d '{"model": "Neural Network", "features": [2699,347,3,0,0,2096,213,234,159,6853,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "decode" : true}' http://127.0.0.1:5000/predict
```

returns
```json
{"prediction":"Spruce/Fir"}
```
