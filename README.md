# Housing price predictor

A production-ready housing price predictor built in Apache Spark.

## Usage steps

1. Make sure to have [poetry installed](https://python-poetry.org/docs/), [git-lfs](https://git-lfs.com/) and Docker installed.
2. Run `git lfs pull` to ensure you have all the files in place.
3. Run `make start-server`
4. Access [http://localhost:8080/docs](http://localhost:8080/docs) to check the endpoint documentation.
5. Import the collection [house_price_predictor.postman_collection.json](house_price_predictor.postman_collection.json) in [Postman](https://www.postman.com/) to test the API. You can try removing features or adding noise to them.

## Project Walk-through

1. In the [data viz notebook](00_data_viz.ipynb), you will find initial visualizations that I performed on each feature.
2. In the [feature engineering notebook](01_feat_eng.ipynb), I first explain how the new features were generated. Then, I overview the construction of the Model Pipeline that is later used to train and put a model in production.
3. In the [model and evaluation notebook](02_model_and_eval.ipynb), the model evaluation process is defined. Then, it contains the hyperparameter-tuning process and finally persisting the winning Pipeline Model so that the production server can use it.
4. In the Python [server module](house_price_predictor/server/main.py), we define a FastAPI endpoint (the `predict()` PUT method). It receives a list of rows according to the schema of the input dataset we have worked on in this project. It loads from the disk the Pipeline that was persisted in the previous model and evaluation notebook, and then applies it to the batch of predictions.
5. The [Dockerfile](Dockerfile) has the following steps:
   1. Installs Java to be able to run Spark.
   2. Install Python.
   3. Installs every dependency in this [poetry project](pyproject.toml).
   4. Copies the source code.
   5. Copies the Pipeline Model.
   6. Starts the server.

## Production Model

### Noisy/Missing data

One of the criteria is that the service should be able to handle noisy data and missing values.
The server application turns any invalid feature (e.g., a float with "3.%0") into a null and then the ML Pipeline performs imputation of every null with the feature average.
These averages were computed with the training set.

For every invalid feature in a data row of the request, the response will provide a description, but it will also provide along a list of `warnings` mentioning every feature that was imputed.

### Keyed predictions

An ML API should always match its predictions with unique keys/IDs.
This because if we want to do distributed processing in the background, it might be hard to ensure the response is in the same order in that the data was received.
This way, we leave the responsibility of matching the provided keys/IDs to the client and won't cause breaking changes whenever we can no longer provide ordering guarantees.
