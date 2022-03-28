FROM continuumio/miniconda3

WORKDIR /build

COPY environment.yml /build/

RUN conda env create --file environment.yml \
    && mkdir -p /home/app

COPY . /home/app

WORKDIR /home/app

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "streamlit", "/bin/bash", "-c"]

EXPOSE 8501

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "streamlit", "streamlit", "run", "./src/bluebook-bulldozer/app.py"]
