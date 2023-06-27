FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN mkdir -p /app

COPY ./app /app
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

ENV PORT 8001

EXPOSE $PORT

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload
