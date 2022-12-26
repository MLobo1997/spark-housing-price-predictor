FROM ubuntu:lunar-20221207

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Install python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


#Install python dependencies
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Create user
WORKDIR /home/myuser
RUN adduser --disabled-password myuser
ENV HOME=/home/myuser

# Add data
ADD models/ models/

# Add src code and start server
ADD house_price_predictor/ house_price_predictor/

RUN chown -R myuser /home/myuser
USER myuser

CMD ["uvicorn", "house_price_predictor.server.main:app", "--host", "0.0.0.0", "--port", "8080"]
#Use this one for heroku
#CMD uvicorn server.main:app --app-dir src --host 0.0.0.0 --port ${PORT##\\}
