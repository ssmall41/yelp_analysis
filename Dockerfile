# Build a docker image for the yelp data set
FROM ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev nano \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip install numpy==1.14.3 \
  && pip install pandas==0.22.0 \
  && pip install scikit-learn==0.19.2 \
  && pip install scipy==1.1.0 \
  && pip install ipython==6.1.0

ADD models.py /yelp/
ADD data/ yelp/data/
WORKDIR /yelp/


#ENTRYPOINT ["python3"]

