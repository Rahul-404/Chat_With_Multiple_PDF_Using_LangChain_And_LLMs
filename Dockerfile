FROM python:3.10.15

COPY . .

CMD [ "pip", "install", "-r", "requirements.txt" ]

# Expose port for the Streamlit app (default is 8501)
EXPOSE 8501

CMD [ "streamlit", "run", "app.py" ]

