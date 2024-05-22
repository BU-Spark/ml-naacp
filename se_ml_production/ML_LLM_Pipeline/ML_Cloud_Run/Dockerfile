FROM python:3.10.0

WORKDIR /code

COPY ./Cloud_Run/requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8080

COPY ./Cloud_Run ./app

ENV GOOGLE_APPLICATION_CREDENTIALS=/code/app/keys/naacp-ser-acc.json

ENV PYTHONPATH "${PYTHONPATH}:/code/app"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
