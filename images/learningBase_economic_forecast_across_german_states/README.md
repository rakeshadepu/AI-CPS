---

## Learning Base – AI-Based Economic Forecast Across German States

---

## 1. Project Ownership

**Project Team Members**

* Rakesh Adepu
* Rohith Boggula

This repository is a **fork of the official AI-CPS repository by Marcus Grum** and has been modified to realize an **individual AI project** in accordance with the course specifications.

---

## 2. Academic Context

This Docker image was created as part of the course:

**“M. Grum: Advanced AI-based Application Systems”**

offered by the
**Junior Chair for Business Information Science, esp. AI-based Application Systems**,
**University of Potsdam**, Germany.

The image represents the **Learning Base** of an AI-based Economic Forecast Across German States.

---

## 3. Purpose of the Learning Base

The **Learning Base** provides all datasets required for:

* AI model **training**
* AI model **testing / validation**

It is strictly separated from the Activation Base to ensure:

* reproducibility
* clean AI-CPS architecture
* correct separation between learning and application phases

---

## 4. Docker Image Information

* **Docker Image Name:**
  `learningbase_economic_forecast_across_german_states`
* **Base Image:**
  `busybox`

This image contains **only two elements**:

1. Preprocessed learning datasets
2. This `READEME.md` file

---

## 5. Dataset Files & Paths (Mandatory Specification)

Inside the container, the following files are provided:

```text
/tmp/learningBase/train/training_data.csv
/tmp/learningBase/validation/test_data.csv
```

These paths are **fixed** and must be used by any AI training or evaluation service.

---

## 6. Dataset Description

Both datasets are **fully preprocessed and model-ready**.

### Features Included

* `state` – German federal state
* `year` – observation year
* `population` – log-transformed & normalized
* `employment` – log-transformed & normalized
* `gdp` – log-transformed & normalized (target variable)

 No further preprocessing must be applied during training or testing.

---
Here is a **clean, exact replacement** for **Section 7: Origin of the Data**, rewritten to **explicitly name the correct source** and suitable for **both Learning Base and Activation Base Readme.md files**.

You can **copy–paste this section directly** and replace the old one.

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

Any derivative work must comply with this license.

---

## 9. Testing the Image

The image can be tested using a `docker-compose.yml` file with an external volume:

This mounts the internal `/tmp` directory for inspection.

---

## 10. Publication

This image is intended to be published on a **public Docker Hub profile**.

---
