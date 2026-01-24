FROM python:3.10-slim

WORKDIR /app

# Copy inference scripts
COPY activation_ann.py .
COPY activation_ols.py .
COPY README.md .

# Install required libraries
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    tensorflow-cpu \
    statsmodels \
    scikit-learn

CMD ["sh"]
