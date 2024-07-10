# Klasyfikacja wybranych komórek szpiku kostnego na podstawie zdjęć rozmazów przy użyciu algorytmu opartego na splotowych sieciach neuronowych

> Projekt inżynierski Mateusz Woźniak


Celem pracy jest porównanie kilku algorytmów uczenia maszynowego z wykorzystaniem splotowych sieci neuronowych do rozpoznawania typów komórek na podstawie zdjęć przedstawiających obraz mikroskopowy rozmazu szpiku kostnego.
Praca zawiera algorytm ekstrakcji obrazów komórek z dużego skanu rozmazu.

![app](https://github.com/matisiekpl/thesis/assets/21008961/1917d3d0-a812-41b0-8ffe-ff9bf3bec38e)

## Uruchomienie

Poniższe komendy uruchomią proces trenowania sieci neuronowej.

```bash
git clone https://github.com/matisiekpl/thesis
pip install flask flask-cors torch torchvision tqdm seaborn scikit-learn pandas numpy matplotlib grad-cam
python train.py
python predict.py
```
