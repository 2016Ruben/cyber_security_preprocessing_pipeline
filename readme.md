# Requirements

Used modules as we have them installed on our system:

matplotlib==3.8.1
PyYAML==5.4.1
scikit-learn==1.3.2
seaborn==0.13.0
tensorflow==2.15.0

The full list can be found in requirements.txt.

# How to hook into Kitsune

## Requirements

We have used additionally:
- Cython==3.0.8
- scapy==2.5.0

## How to install

1. Copy the [Kitsune repository](https://github.com/ymirsky/Kitsune-py) into kitsune. We used commit 28a654b5813936380d264c0934136efda672174a
2. Rename directory into "Kitsune" (most likely named "Kitsune-py" before)
3. Kitsune is now in a directory kitsune/Kitsune

## On Linus

If you're using Linux, then you can simply execute the following commands:

```
cd kitsune
git clone https://github.com/ymirsky/Kitsune-py
mv Kitsune-py Kitsune && cd Kitsune
git checkout 28a654b5813936380d264c0934136efda672174a
```

