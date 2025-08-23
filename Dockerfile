# base image
FROM python:3.12-slim

# setting a working directory
WORKDIR /app

#copying requirements and installing dependeces
COPY requirements.txt .
RUN pip install -r requirements.txt

#copying rest of the code
COPY ./data/transformed ./data/transformed
COPY ./second_hand_car_price_prediction/Frontend ./second_hand_car_price_prediction/Frontend
    
#exposing port
EXPOSE 8501 

#command to start streamlit application
CMD ["streamlit", "run" ,"./second_hand_car_price_prediction/Frontend/main.py","--server.port=8501", "--server.address=0.0.0.0"]