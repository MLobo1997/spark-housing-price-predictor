start-fast-api:
	poetry install
	poetry export --without-hashes -f requirements.txt --output requirements.txt
	docker build -t miguelobo/house-price-predictor:0.0.1 .
	docker stop house-price-predictor; docker rm house-price-predictor; docker run --expose 8080 -p 8080:8080 --name house-price-predictor miguelobo/house-price-predictor:0.0.1
