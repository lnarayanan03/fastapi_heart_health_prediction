FROM python:3.12.4
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./heart.csv /code/
COPY ./generate_model.py /code/generate_model.py
COPY ./app /code/app
RUN python generate_model.py
RUN mv best_model_heart_prediction.pkl app/best_model_heart_prediction.pkl
RUN rm /code/heart.csv
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# FROM python:3.13
# WORKDIR /code
# COPY ./requirements.txt /code/requirements.txt
# RUN pip install -r /code/requirements.txt
# COPY ./housing.csv /code/
# COPY ./generate_model.py /code/generate_model.py
# COPY ./app /code/app
# RUN python generate_model.py
# RUN mv rfr_v1.pkl app/rfr_v1.pkl
# RUN rm /code/housing.csv
# CMD ["fastapi", "run", "app/main.py", "--port", "80"]
