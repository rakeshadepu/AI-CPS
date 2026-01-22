---

## Activation Base – AI-Based Economic Forecast Across German States

---

## 1. Project Ownership

**Project Team Members**

* Rakesh Adepu
* Rohith Boggula

This repository is a **fork of the official AI-CPS repository by Marcus Grum**, adapted to realize an individual AI application in compliance with course requirements.

---

## 2. Academic Context

This Docker image was created as part of the course:

**“M. Grum: Advanced AI-based Application Systems”**

offered by the
**Junior Chair for Business Information Science, esp. AI-based Application Systems**,
**University of Potsdam**, Germany.

This image represents the **Activation Base** of the AI-CPS pipeline.

---

## 3. Purpose of the Activation Base

The **Activation Base** provides the data required for:

* AI model **application**
* AI model **deployment**
* AI inference on unseen data

It contains **no training or testing data** and is strictly isolated from the Learning Base.

---

## 4. Docker Image Information

* **Docker Image Name:**
  `activationbase_economic_forecast_across_german_states`
* **Base Image:**
  `busybox`

The image contains:

1. Activation dataset
2. This `README.md` file

---

## 5. Dataset File & Path (Mandatory Specification)

Inside the container, the following file is provided:

```text
/tmp/activationBase/activation_data.csv
```

This dataset represents a **single unseen data sample** used for AI system activation.

---

## 6. Dataset Description

The activation dataset contains **one fully preprocessed row**, derived from the test dataset.

### Included Features

* `state` – German federal state
* `year` – observation year
* `population` – log-transformed & normalized
* `employment` – log-transformed & normalized
* `gdp` – log-transformed & normalized

 All transformations were applied **before** containerization.

---

## 7. Origin of the Data

The data used in this project originates from the **German Federal Ministry of Research, Technology and Space** (*Bundesministerium für Forschung, Technologie und Raumfahrt*).

The datasets include officially published statistics on:

* Population by German federal state
* Employment (employed persons) by German federal state
* Gross Domestic Product (GDP) by German federal state

The raw data was **scraped from publicly accessible governmental data portals** maintained by the Federal Ministry of Research, Technology and Space and subsequently **cleaned, transformed, and preprocessed** for use in this project.

All data processing was performed **exclusively for academic and educational purposes** within the scope of the course *“M. Grum: Advanced AI-based Application Systems”* at the University of Potsdam.

---

## 8. License Commitment

This Docker image and its contents are released under the:

**GNU Affero General Public License v3.0 (AGPL-3.0)**

---

## 9. Testing the Image

The image can be tested using an external Docker volume:

This allows inspection of the activation dataset.

---

## 10. Publication

This image is intended to be published on a **public Docker Hub profile**.

---
