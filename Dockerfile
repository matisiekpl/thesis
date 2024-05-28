FROM node:21 as builder
WORKDIR /app
COPY app/ .
RUN ls 
RUN yarn install
RUN yarn build

FROM python
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install flask flask-cors torch torchvision tqdm seaborn scikit-learn pandas numpy matplotlib grad-cam
EXPOSE 5001
COPY --from=builder /app/dist /app/dist
CMD ["python", "/app/serve.py"]