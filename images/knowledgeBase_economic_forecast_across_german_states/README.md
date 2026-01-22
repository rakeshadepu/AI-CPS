---

## Knowledge Base – AI-Based Economic Forecast Across German States

---

## 1. Project Ownership

**Project Team Members**

* Rakesh Adepu
* Rohith Boggula

This project is an individual realization based on the official **AI-CPS reference architecture by Marcus Grum**, developed to fulfill the requirements of the course.

---

## 2. Academic Context

This Docker image was created as part of the course:

**“M. Grum: Advanced AI-based Application Systems”**

offered by the
**Junior Chair for Business Information Science, esp. AI-based Application Systems**,
**University of Potsdam**, Germany.

This image represents the **Knowledge Base** component of the AI-CPS pipeline.

---

## 3. Purpose of the Knowledge Base

The **Knowledge Base** contains the trained intelligence of the AI system.

It provides:

* A fully trained AI / OLS forecasting model
* Persisted model parameters for inference
* No executable code and no raw data

The Knowledge Base is strictly separated from the Learning Base and Activation Base.

---

## 4. Docker Image Information

* **Docker Image Name:**
  `knowledgebase_economic_forecast_across_german_states`
* **Base Image:**
  `busybox`

The image contains:

1. The trained AI model file
2. This `README.md` file

---

## 5. Model File & Path (Mandatory Specification)

Inside the container, the trained model is available at:

```text
/tmp/knowledgeBase/currentAiSolution.h5
```

---

## 6. Model Description

The stored model represents a **regression-based AI forecasting system** trained to predict:

* Gross Domestic Product (GDP) of German federal states

### Model Characteristics

* Input features:

  * Population
  * Employment
  * Year
  * Federal state 
* Output:

  * Normalized GDP 
* Training:

  * Conducted using preprocessed historical data
  * Feature scaling and transformations applied before training

The model is stored in **serialized form** to ensure reproducibility and portability.

---

## 7. Origin of the Data

The model was trained using datasets originating from the
**German Federal Ministry of Research, Technology and Space**
(*Bundesministerium für Forschung, Technologie und Raumfahrt*).

All data processing, training, and evaluation were performed **exclusively for academic and educational purposes** within the scope of the course.

---

## 8. License Commitment

This Docker image and its contents are released under the:

**GNU Affero General Public License v3.0 (AGPL-3.0)**

---

## 9. Publication

This image is intended to be published on a **public Docker Hub profile**.

---