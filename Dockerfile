FROM python
WORKDIR /app
COPY . .
RUN pip install flask flask-cors torch torchvision tqdm seaborn scikit-learn pandas numpy matplotlib
EXPOSE 5001
CMD ["python", "/app/serve.py"]