
## code Base – AI-Based Economic Forecast across German Federal States

---

## 1. Project Ownership

**Project Team Members**

* Rakesh Adepu
* Rohith Boggula

This repository is a **fork of the official AI-CPS repository by Marcus Grum**, adapted to implement an individual AI application in full compliance with the course requirements.

---

## 2. Academic Context

This Docker image was created as part of the course:

**“M. Grum: Advanced AI-based Application Systems”**

offered by the
**Junior Chair for Business Information Science, esp. AI-based Application Systems**,
**University of Potsdam**, Germany.

This image represents the **Code Base** of the AI-CPS pipeline.

---

## 3. Purpose of the Code Base

The **Code Base** contains the **executable inference logic** required to:

* Load **pre-trained and persisted models**
* Apply these models to **previously unseen activation data**
* Produce **GDP predictions** during system activation

This image **does not contain training or testing logic** and is strictly separated from:

* Learning Base (model training)
* Knowledge Base (model storage)

---

## 4. Docker Image Information

* **Docker Image Name:**
  `codebase_economic_forecast_across_german_states`

* **Base Image:**
  `busybox`

---

## 5. Contents of the Code Base

The Code Base contains the following files:

### 5.1 Inference Scripts

* **`activation_ann.py`**
  Loads the saved **Artificial Neural Network (ANN)** model
  (`currentAiSolution.keras`) and performs GDP prediction using the activation dataset.

* **`activation_ols.py`**
  Loads the saved **Ordinary Least Squares (OLS)** regression model
  (`currentOlsSolution.pkl`) and performs GDP prediction using the activation dataset.

Both scripts are designed **exclusively for inference** and do not modify model parameters.

---

### 5.2 Model Files (Mounted from Knowledge Base)

The following pre-trained models are loaded at runtime:

* `currentAiSolution.keras` – Trained ANN model
* `currentOlsSolution.pkl` – Trained OLS regression model

These models are **not trained inside this image** and are expected to be provided via the Knowledge Base container.

---

### 5.3 Activation Dataset

Inside the container, the activation dataset is available at:

```text
/tmp/codeBase/activation_data.csv
```

This dataset contains **one unseen, fully preprocessed data sample** used to activate the AI system.

---

## 6. Dataset Description

The activation dataset contains **a single data record** derived from the original test dataset.

### Included Features

* `state` – German federal state
* `year` – observation year
* `population` – log-transformed and normalized
* `employment` – log-transformed and normalized
* `gdp` – log-transformed and normalized (target variable)

All preprocessing steps were applied **prior to containerization**.

---

## 7. Functional Role in the AI-CPS Pipeline

The Code Base is responsible for:

* Executing inference code
* Loading persisted models
* Applying activation data
* Producing GDP predictions

It acts as the **final activation layer** of the AI-CPS architecture.

---

## 8. Origin of the Data

The data used in this project originates from the
**German Federal Ministry of Research, Technology and Space**
(*Bundesministerium für Forschung, Technologie und Raumfahrt*).

The raw datasets include official statistics on:

* Population by German federal state
* Employment by German federal state
* Gross Domestic Product (GDP) by German federal state

The data was collected from publicly accessible governmental data portals and subsequently **cleaned, transformed, and preprocessed** for academic use.

---

## 9. License Commitment

This Docker image and its contents are released under the:

**GNU Affero General Public License v3.0 (AGPL-3.0)**

---

## 10. Publication

This image is intended to be published on a **public Docker Hub profile** as part of the AI-CPS project submission.

---
