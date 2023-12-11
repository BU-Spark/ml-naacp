FROM python:3.10.0

WORKDIR /code

COPY ./src/requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8080

COPY ./src ./app

ENV PYTHONPATH "${PYTHONPATH}:/code/app"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
