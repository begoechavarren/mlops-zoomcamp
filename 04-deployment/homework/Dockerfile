FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY predict.py .

ENTRYPOINT ["python", "predict.py"]