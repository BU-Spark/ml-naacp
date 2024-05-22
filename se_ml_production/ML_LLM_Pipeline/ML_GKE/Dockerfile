FROM python:3.10.0

WORKDIR /code

COPY ./ML_Service_GKE/requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8080

COPY ./ML_Service_GKE .

ENV GOOGLE_APPLICATION_CREDENTIALS=/code/keys/naacp-ser-acc.json

ENV PYTHONPATH "${PYTHONPATH}:/code"

CMD ["python", "main.py"]
