# https://hub.docker.com/_/python

FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app
COPY ./app/application /code/app/application


# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]