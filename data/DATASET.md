# Dataset Installation

This project uses the publicly available dataset from the following research paper:

**Real-Time Series Arc Fault Detection and Appliances Classification in AC Networks Based on Competing Convolutional Kernels**

## Dataset Source

Zenodo Repository:

https://zenodo.org/records/15279701

## Download

Download the following dataset file:

```
dataset_2classes_20ms_100kHz.mat
```

This dataset contains:

- Binary classification dataset (Normal / Series Arc)
- Sampling Frequency: **100 kHz**
- Window Length: **20 ms**
- One waveform per sample (2000 current measurements)

## Installation

Place the downloaded file inside the project's `data` directory.

Project structure should look like:

```
arc-fault-project/
│
├── data/
│   └── dataset_2classes_20ms_100kHz.mat
│
├── notebooks/
├── README.md
└── requirements.txt
```
