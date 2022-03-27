FROM continuumio/miniconda:latest

WORKDIR /build
# While iterating on build, copy app/ and/or src/ contents separately AFTER conda setup.
# This will enable docker to cache more layers.
COPY . /build/

RUN mkdir -p /usr/app \
    && mv /build/app /usr/ \
    && pip install /build/src/. \
    && conda install --file requirements.txt \
    && conda clean -y --all \
    && rm -r /build

WORKDIR /usr/app

CMD ["streamlit", "run", "app.py"]
