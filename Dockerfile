FROM python:3

ADD . .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir flask pandas statsmodels matplotlib scikit-learn

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]