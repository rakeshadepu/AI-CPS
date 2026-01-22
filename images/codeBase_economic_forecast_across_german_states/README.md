---

## code Base – AI-Based Economic Forecast across German Federal States

---

## 1. Project Ownership

**Project Team Members**

* Rakesh Adepu
* Rohith Boggula

This repository is a **fork of the official AI-CPS repository by Marcus Grum**, adapted to implement an individual AI application in compliance with the course requirements.

---

## 2. Academic Context

This Docker image was created as part of the course:

**“M. Grum: Advanced AI-based Application Systems”**

offered by the
**Junior Chair for Business Information Science, esp. AI-based Application Systems**,
**University of Potsdam**, Germany.

This image represents the **code Base** of the AI-CPS pipeline.

---

## 3. Purpose of the code Base

The **code Base** provides the data required exclusively for:

* AI model **application**
* AI model **deployment**
* AI inference on **previously unseen data**

It contains **no training or testing data** and is strictly separated from the Learning Base and Knowledge Base.

---

## 4. Docker Image Information

* **Docker Image Name:**
  `codeBase_economic_forecast_across_german_states`
* **Base Image:**
  `busybox`

The image contains:

1. A single activation dataset
2. This `README.md` file

---

## 5. Dataset File & Path (Mandatory Specification)

Inside the container, the activation dataset is available at:

```text
/tmp/codeBase/activation_data.csv
```

This dataset represents a **single unseen data sample** used to activate the AI system.

---

## 6. Dataset Description

The activation dataset contains **one fully preprocessed data record**, derived from the original test dataset.

### Included Features

* `state` – German federal state
* `year` – observation year
* `population` – log-transformed and normalized
* `employment` – log-transformed and normalized
* `gdp` – log-transformed and normalized

All preprocessing and transformations were applied **prior to containerization**.

---

## 7. Origin of the Data

The data used in this project originates from the
**German Federal Ministry of Research, Technology and Space**
(*Bundesministerium für Forschung, Technologie und Raumfahrt*).

The raw datasets include official statistics on:

* Population by German federal state
* Employment by German federal state
* Gross Domestic Product (GDP) by German federal state

The data was scraped from publicly accessible governmental data portals and subsequently **cleaned, transformed, and preprocessed** for academic use.

---

## 8. License Commitment

This Docker image and its contents are released under the:

**GNU Affero General Public License v3.0 (AGPL-3.0)**

---

## 9. Publication

This image is intended to be published on a **public Docker Hub profile**.

---