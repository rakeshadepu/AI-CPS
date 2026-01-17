# AI-Based Economic Forecast Across German States

## Course Information

This project is developed as part of the course
**“M. Grum: Advanced AI-based Application Systems”**
Business Information Systems, esp. AI-based Application Systems
Junior Chair for Business Information Science, esp. AI-based Application Systems
University of Potsdam

---

## Project Ownership

**Team Members:**

* *Rakesh Adepu*
* *Rohith Boggula*

This repository is a **fork of the official AI-CPS repository** by Marcus Grum and has been modified to realize an individual AI project according to the course specifications.

---

## Project Idea

The goal of this project is to build an **AI-based system that analyzes and compares economic growth indicators across German federal states**.

The system:

* Scrapes publicly available economic data from the Internet
* Cleans, normalizes, and prepares the data for training and testing
* Trains an **Artificial Neural Network (ANN)** model and an **OLS regression model**
* Compares AI-based predictions with classical econometric results
* Provides all datasets, models, and applications via **Docker images**

---

## Data Description and Origin

The dataset consists of **economic growth indicators across German states**, such as:

* Gross Domestic Product (GDP)
* Employment levels
* Population statistics
* Year-based economic indicators

**Data Sources (publicly available):**

* Federal Statistical Office of Germany (Destatis)
* Official German government and open-data portals

All data scraping and usage complies with publicly accessible data policies.

---

## AI and OLS Models

### AI Model

* Artificial Neural Network (ANN)
* Implemented using **TensorFlow**
* Trained on 80% of the dataset
* Validated on 20% test data

### OLS Model

* Ordinary Least Squares (OLS) regression
* Implemented using **Statsmodels**
* Solves the same prediction task as the AI model
* Enables direct performance comparison

---

## Docker Images

The project provides the following Docker images (all based on `busybox`):

* `learningBase_ai_economic_growth`
* `activationBase_ai_economic_growth`
* `knowledgeBase_ai_economic_growth`
* `codeBase_ai_economic_growth`

Each image contains:

* Required data or model files
* A `README.md` specifying:

  * Ownership
  * Course reference
  * Data origin
  * Commitment to the **AGPL-3.0 License**

All images are published on **Docker Hub** and can be pulled using documented commands in the report.

---

## Docker-Compose Usage

Docker-compose files are provided to:

* Apply the trained **AI model**
* Apply the trained **OLS model**
* Use a shared external volume `ai_system` mounted at `/tmp`

This ensures consistent execution across all components.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

All reused code from the AI-CPS repository remains under its original license.

---

## Reproducibility Statement

This repository contains **all necessary code, documentation, and configuration files** to:

* Rebuild the datasets
* Retrain the AI and OLS models
* Reproduce the experimental results
* Re-run the system using Docker and Docker Compose

---

## Course Reference

This repository is **explicitly part of the course**
**“M. Grum: Advanced AI-based Application Systems”**
at the **University of Potsdam**.

---

