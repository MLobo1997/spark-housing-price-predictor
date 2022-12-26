# Housing price predictor


## Production Model

### Noisy/Missing data

One of the criteria of the assignment is that the service should be able to handle noisy data and missing values.
The server application turns any invalid feature (e.g., a float with "3.%0") into a null and then the ML Pipeline performs imputation of every null with the feature average.
These averages were computed with the training set.

### Testing

If you wish to test the application, I have added the collection [house_price_predictor.postman_collection.json](house_price_predictor.postman_collection.json) which you can simply import in 
[Postman](https://www.postman.com/) to test the API. You can try removing features or adding noise to them.

## Further steps
