# Module 1: Bank Marketing Model - Complete ML Workflow on Cloudera AI

Welcome to Module 1! In this lab, you'll build a complete machine learning pipeline on Cloudera AI, from data ingestion through model deployment and inference. This module demonstrates real-world ML workflows using industry-standard tools and best practices.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Lab Structure](#lab-structure)
4. [Step-by-Step Guide](#step-by-step-guide)
   - [Step 1: Data Ingestion](#step-1-data-ingestion)
   - [Step 2: Exploratory Data Analysis](#step-2-exploratory-data-analysis)
   - [Step 3: Model Training](#step-3-model-training)
   - [Step 4: Model Deployment](#step-4-model-deployment)
   - [Step 5: Inference Pipeline](#step-5-inference-pipeline)
   - [Step 6: Inference Deep Dive](#step-6-inference-deep-dive)
5. [Understanding the Data Flow](#understanding-the-data-flow)
6. [Troubleshooting](#troubleshooting)

---

## 📊 Understanding ML Evaluation Metrics (Prerequisites)

Before diving into model training, familiarize yourself with the metrics you'll be tracking:

[📖 View Presentation: ML Evaluation Metrics](../assets/ML_Evaluation_Metrics_Final.pdf)

**Key metrics you'll encounter in this lab:**
- **F1 Score** - Balance between precision and recall (important for imbalanced datasets)
- **ROC-AUC** - Model's ability to distinguish between classes at various thresholds
- **Accuracy** - Overall correctness (but can be misleading with imbalanced data)
- **Precision** - Of predicted positives, how many were actually correct
- **Recall** - Of actual positives, how many did we catch

Each metric tells a different story about your model's performance. Understanding these is essential before evaluating your models in Steps 3 and 4.

---

## Overview

This lab walks you through building a **bank marketing prediction model** that classifies whether a customer will subscribe to a term deposit. You'll learn how to:

- **Ingest raw data** into a data lake
- **Explore and analyze** data with interactive notebooks
- **Train multiple models** using MLflow experiment tracking
- **Deploy models** as API endpoints with Cloudera AI
- **Process inference data** with feature engineering pipelines
- **Make predictions** and track results in production

**Real-World Context:** This workflow mirrors how data scientists work in production environments, where data comes in, gets processed, models make predictions, and results feed downstream systems.

---

## Lab Structure

The lab is organized as a numbered sequence of scripts and notebooks:

| Step | File | Type | Purpose |
|------|------|------|---------|
| 1 | `01_ingest.py` | Python Script | Load raw data into the data lake |
| 2 | `02_eda_notebook.ipynb` | Jupyter Notebook | Explore and visualize data patterns |
| 3 | `03_train_quick.py` | Python Script | Train models with MLflow tracking |
| 4 | `04_deploy.py` | Python Script | Deploy best model as API endpoint |
| 5a | `05.1_inference_data_prep.py` | Python Script | Engineer features for inference |
| 5b | `05.2_inference_predict.py` | Python Script | Generate predictions from new data |
| 6 | `06_Inference_101.ipynb` | Jupyter Notebook | Interactive inference exploration |

Execute scripts in order. Each step builds on previous outputs.

---

## Step-by-Step Guide

### Pre Setup - 

Follow the instructors guide to get you to this portal page:

![main-page](images/cai-portal.png)


Click into CAI. 
Find your project and click into it. 

The first thing you'll want to do is make yourself the project owner - follow the directions show below.

![project-owner](images/update-project-owner.png)



Open a session as follows. Make sure

![session-details](images/session-details.png)

Make sure you enable spark and pick the spark 3.3.0 addon option.

Once in the session, open up the terminal as shown below: 

**Note** Througth out this lab always check you are in the right module folder. If you get a failure, check this first!


![session-details](images/terminal.png)


### Step 1: Data Ingestion

**Purpose:** Load raw bank marketing data and prepare it for analysis and training.

**What Happens:**
1. Reads customer banking data from a CSV source
2. Creates a sample inference dataset (test customers)
3. Saves data to the Cloudera AI data lake
4. Validates data schema and quality

**To Run:**
```bash
cd module1
python 01_ingest.py
```

**Expected Output:**
```
Data loaded: 41,188 customer records
Inference sample created: 1,000 test customers
✓ Data saved to: data/bank-additional/
✓ Inference data saved to: inference_data/raw_inference_data.csv
```

**What You're Creating:**
- **`data/bank-additional/bank-additional-full.csv`** - Complete training dataset (41,188 rows)
  - Features: Age, job, marital status, education, account balance, campaign details, economic indicators
  - Target: Whether customer subscribed to term deposit (yes/no)
- **`inference_data/raw_inference_data.csv`** - Sample inference data (1,000 rows)
  - Same features as training data
  - Used to simulate real-world prediction scenarios
  - Will be processed through the inference pipeline

**Why This Matters:**
In production, raw data constantly flows into your data lake. This step simulates that process—you're setting up your data foundation for everything downstream.

---

### Step 2: Exploratory Data Analysis

**Purpose:** Understand data patterns, distributions, and relationships before building models.

**What Happens:**
The Jupyter notebook `02_eda_notebook.ipynb` provides:
- Statistical summaries of features
- Distribution visualizations
- Correlation analysis
- Missing value assessment
- Business insights about customer behavior

**To Run:**
```bash
# Open the notebook in your Cloudera AI project
# Click on: 02_eda_notebook.ipynb
# Run all cells to see the analysis
```

**Key Insights You'll Discover:**
- Customer age distribution and campaign patterns
- Economic indicator trends
- Class imbalance in the target variable (important for training)
- Feature correlations with subscription likelihood

**Why This Matters:**
EDA informs your entire ML pipeline. Understanding your data helps you:
- Identify which features matter most
- Spot data quality issues early
- Make informed decisions about preprocessing
- Validate if your model results make business sense

---

### Step 3: Model Training

**Purpose:** Train multiple models using MLflow to track experiments and find the best performer.

**What We're Doing (In the Interest of Time):**
For this lab, we're using **`03_train_quick.py`** which trains a focused set of models quickly (~2-5 minutes). This version:
- Tests 1 model type (Logistic Regression) with multiple configurations
- Tests both baseline and engineered features
- Handles class imbalance with SMOTE
- Logs all experiments to MLflow

*Note: `03_train_extended.py` is available if you want to experiment with more model types (Random Forest, XGBoost, SVM, etc.) - it takes 15-30 minutes.*

**To Run:**
```bash
python 03_train_quick.py
```

**Expected Output:**
```
==================== EXPERIMENT TRACKING ====================
Experiment: bank_marketing_experiments
Tracking URI: ...

✓ Run 1: Baseline features (SMOTE enabled)      | F1: 0.512
✓ Run 2: Engineered features (SMOTE enabled)    | F1: 0.521
✓ Run 3: Baseline features (SMOTE disabled)     | F1: 0.495
✓ Run 4: Engineered features (SMOTE disabled)   | F1: 0.508

Total runs completed: 8
===============================================================
```

**What Gets Logged:**
- Model parameters (regularization, solver, etc.)
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Training time and data shapes
- Model artifacts (the actual trained model file)

![experiments](images/experiments-1.png)

![experiments](images/experiments-2.png)

**Next: Explore in Cloudera AI**
After running the training script:
1. Go to your Cloudera AI project
2. Click on **"MLflow Experiments"** or **"Experiments"** tab
3. Select **"bank_marketing_experiments"**
4. You'll see all 8 experiment runs with their metrics
5. Compare F1 scores, ROC-AUC, and other metrics
6. Observe which feature engineering approach performs best

**Why This Matters:**
MLflow is how data scientists track experiments at scale. In production:
- You run hundreds of experiments
- You need to compare results systematically
- You track which models were trained on which data
- You can reproduce any experiment later
- You select the best model for deployment

---

### Step 4: Model Deployment

**Purpose:** Take the best trained model and deploy it as an API endpoint so applications can request predictions.

**The Deployment Process:**
The deployment script (`04_deploy.py`) handles these steps automatically:

1. **Select Best Model**
   - Queries MLflow experiments
   - Finds the model with highest F1 score
   - Retrieves preprocessing artifacts

2. **Register the Model**
   - Saves model in MLflow Model Registry
   - Makes it available for deployment
   - Tracks model version and metadata

3. **Build Model Service**
   - Packages model with dependencies
   - Creates API endpoint configuration
   - Sets up monitoring and logging

4. **Deploy to Cloudera AI**
   - Registers model service in your project
   - Model becomes available as REST API
   - Can be called by applications for predictions

**To Run:**
```bash
python 04_deploy.py
```

**Expected Output:**
```
==================== MODEL DEPLOYMENT ====================
Best Model: bank_marketing_experiments (Run: abc123xyz)
Metrics: F1 Score: 0.521, ROC-AUC: 0.943

✓ Model registered in MLflow Model Registry
✓ Model packaged and ready for deployment
✓ Deploying to Cloudera AI...

Deployment Status: SUCCESS
API Endpoint: https://modelservice.ai.project.com/model
Access Key: [your-api-key]

Model is now live! Ready to receive prediction requests.
============================================================
```

**Next: Verify Deployment in Cloudera AI**
1. Go to your project's **Models** or **Deployments** tab
2. Find the deployed model (bank_marketing_model)
3. Check the status (should be "Running" or "Healthy")
4. Note the API endpoint URL
5. View logs and monitoring information

**Why This Matters:**
This is where ML moves from experimental to operational:
- Your model is now a production service
- Other applications can query it for predictions
- You can monitor performance, response times, and errors
- You can update or roll back models safely

---

### Step 5: Inference Pipeline

**Purpose:** Build an automated workflow that takes new customer data, prepares it, and generates predictions at scale.


⚠️ **Pre Step - Check** ⚠️
⚠️ **Project Runtime Check** ⚠️

Lets make sure that we have the runtime available to the project scope to create this Job (and the application in module 3). Follow the following steps to check for PBJ runtime, if not add it. 

Ensuring you within you project scope, click on "**Project Settings**"

![experiments](images/runtime-1.png)

 click on "**Runtime**"
 
![experiments](images/runtime-2.png)

Check that both runtimes are present, if not

![experiments](images/runtime-3.png)

click "**add runtime**" then 
add the runtime then 
click "**submit**"

![experiments](images/runtime-4.png)




⚠️ **IMPORTANT NOTE - Lab Environment Only:**
```
In this lab, we use the TRAINING DATA for inference demonstration to show you how the model
responds and to validate the pipeline works end-to-end.

IN PRODUCTION, you would NEVER feed training data back to a deployed model. This is lab-only
practice to help you understand:
  • How data flows through the preprocessing pipeline
  • How the model generates predictions
  • How to interpret results and validate model behavior

Real-world inference uses completely NEW, unseen data from actual customers or business
scenarios. Training data is for model development only.
```

**The Two-Job Pattern (Simulating Real Production):**

In production, inference is typically broken into stages:

#### Job 1: Data Preparation (`05.1_inference_data_prep.py`)
**What it does:**
- Loads raw inference data (new customers to score)
- Applies feature engineering (same transformations as training)
  - Creates engagement scores
  - Generates age groups and economic categories
  - Creates duration categories
- Applies preprocessing (scaling and encoding)
- Outputs engineered data ready for the model

**Output File:**
```
inference_data/engineered_inference_data.csv
├── All 1,000 test customers
├── Original features (age, duration, campaign, etc.)
├── Engineered features (engagement_score, age_group, etc.)
└── Scaled and one-hot encoded for model input
```

#### Job 2: Generate Predictions (`05.2_inference_predict.py`)
**What it does:**
- Loads the engineered data from Job 1
- Loads the best trained model from MLflow
- Makes predictions on all records
- Generates probability scores
- Saves results with row tracking

**Job Creation through UI:**

We are going to create 2 jobs from 2 seperate python script files and we're link a dependency as follows:

### before getting started

We need to update the config file with model details so that various scripts can collect the right model informaiton

![job-create-ui](images/gather-model-details.png)

![job-create-ui](images/paste-model-details.png)




**Click on 'Jobs'**

![job-create-ui](images/job-create-1.png)

**Create New Job**

![job-create-ui](images/job-create-2.png)

**use the file path shown for first job.**

**Then select manual.**

![job-create-ui](images/job-create-3.png)


![job-create-ui](images/job-create-4.png)
### Now Lets Create the second job
![job-create-ui](images/job-create-5.png)

**Finally, Create a dependency to the first job**

![job-create-ui](images/job-create-6.png)


you'll see the dependencies of jobs set up. 

![job-create-ui](images/job-dependency.png)


finally you can run by hitting '**run as** for the first (inference data prep job)

```
**Output File:**
```
inference_data/predictions.csv
├── row_id: Unique identifier for each customer
├── prediction: 0 (won't subscribe) or 1 (will subscribe)
├── probability_class_0: Confidence for "no" prediction
├── probability_class_1: Confidence for "yes" prediction
└── prediction_label: Human-readable label (no/yes)
```

**[PLACEHOLDER FOR IMAGE: Diagram showing data flow through the pipeline with sample rows at each stage]**

**Data Flow Through the Pipeline:**

```
Raw Data (1,000 customers)
         ↓
    [Job 1: Prepare]
    • Apply feature engineering
    • Engineered features created
    • Data scaled and encoded
         ↓
Engineered Data (1,000 customers)
         ↓
    [Job 2: Predict]
    • Load trained model
    • Generate predictions
    • Score probabilities
         ↓
Predictions (1,000 predictions + scores)
```

**Understanding the Output Files:**

**⚠️ Lab Note:** In this lab, `raw_inference_data.csv` contains records sampled from the training dataset for demonstration purposes. In production, inference data would come from completely separate sources (new customers, recent transactions, etc.) that the model has never seen during training.

1. **`inference_data/raw_inference_data.csv`** (Input)
   - Original customer features as they arrive
   - No feature engineering yet
   - This is what a production system would feed in
   - *In this lab: sampled from training data for demo purposes*

2. **`inference_data/engineered_inference_data.csv`** (Intermediate)
   - Same customers after feature engineering
   - New columns: engagement_score, age_group, etc.
   - Scaled and encoded for model input
   - Shows what transformations were applied

3. **`inference_data/predictions.csv`** (Output)
   - Final predictions ready for business use
   - Includes confidence scores
   - Traceable back to original customers via row_id
   - This is what goes to downstream systems

**Why This Matters:**
In production:
- Inference often runs on schedules (hourly, daily, weekly)
- Data might come from multiple sources
- You need to transform new data exactly like training data
- Results feed into business applications
- You track data lineage to maintain reproducibility

---

### Step 6: Inference Deep Dive

**Purpose:** Explore the inference process interactively and understand how the model makes predictions.

⚠️ **Note:** This notebook demonstrates inference using a single record from the training data for educational purposes. In production, you'd use completely new, unseen customer data.

**Interactive Exploration:**

The Jupyter notebook `06_Inference_101.ipynb` takes you through:

1. **Loading a Single Prediction**
   - Loads one test customer record
   - Shows original features
   - Shows engineering steps

2. **Feature Engineering**
   - See how raw features are transformed
   - Understand engagement score calculation
   - View categorical encodings

3. **Preprocessing**
   - Observe scaling applied to numeric features
   - See one-hot encoding for categorical features
   - Compare before/after values

4. **Model Prediction**
   - Send features to the model
   - Receive prediction and probabilities
   - Interpret confidence scores

**To Run:**
```bash
# Open in Cloudera AI:
# Click: 06_Inference_101.ipynb
# Run all cells to walk through inference step-by-step
```

**What You'll Learn:**
- How to prepare data for a trained model
- What happens inside the model
- How to interpret prediction scores
- How to handle edge cases

**A Note About the Model Service:**

The model you deployed in Step 4 is hosted on Cloudera AI within your project. It's a REST API that handles:
- Request validation
- Feature preprocessing
- Model inference
- Response formatting
- Error handling
- Monitoring

In a real deployment, applications would query this endpoint continuously. Later in the lab (or in advanced modules), you'll explore Cloudera AI's inference service capabilities in more detail, including:
- Real-time endpoint monitoring
- A/B testing different models
- Batch prediction jobs
- Model versioning and rollback

### Your Model Endpoint

Every deployed model has its own **endpoint URL** and **access key**.

- `05.2_inference_predict.py` looks these up automatically from the model you deployed in `04_deploy.py`.
- `06_Inference_101.ipynb` prints them and asks you to paste them into the notebook, so you see exactly what a client needs to call the model.

You can also find both in the UI: **Models** tab → **banking_campaign_predictor** → **Overview** → *"Instructions for importing requests"*.

---

## Understanding the Data Flow

Here's the complete journey of data through this lab:

```
┌─────────────────────────────────────────────────────────────┐
│                    STEP 1: DATA INGESTION                   │
│  Raw CSV → Validate → Save to Data Lake                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    ↓                             ↓
┌──────────────────────┐  ┌──────────────────────┐
│  TRAINING DATA       │  │  INFERENCE DATA      │
│  41,188 rows         │  │  1,000 rows          │
│  (bank customers)    │  │  (test customers)    │
└──────────────────────┘  └──────────┬───────────┘
    │                                │
    ↓                                ↓
┌──────────────────────┐      ┌──────────────────────┐
│ STEP 2: EDA          │      │ STEP 5A: ENGINEER    │
│ Explore patterns     │      │ Apply transformations│
│ Visualize data       │      │ Create features      │
└──────────────┬───────┘      └──────────┬───────────┘
               │                         │
               ↓                         ↓
        ┌──────────────┐         ┌──────────────────────┐
        │ Insights     │         │ ENGINEERED DATA      │
        │ • Patterns   │         │ 1,000 rows           │
        │ • Imbalance  │         │ (with new features)  │
        │ • Key vars   │         └──────────┬───────────┘
        └──────────────┘                    │
                                           ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: MODEL TRAINING                         │
│  Baseline Features → Engineered Features → Compare Results  │
│  Test different configurations → Track with MLflow          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
        ┌─────────────────────────┐
        │ 8 EXPERIMENT RUNS       │
        │ Compare F1, ROC-AUC     │
        │ Select best model       │
        └──────────────┬──────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 4: MODEL DEPLOYMENT                       │
│  Register Best Model → Package → Deploy to Cloudera AI      │
│  Result: REST API Endpoint Ready for Predictions            │
└──────────────────┬──────────────────────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    ↓                             ↓
┌──────────────────────┐  ┌──────────────────────┐
│ STEP 5B: PREDICT     │  │ STEP 6: DEEP DIVE    │
│ Load model           │  │ Interactive notebook │
│ Score 1,000 records  │  │ Explore predictions  │
│ Generate output      │  │ Understand features  │
└──────────────┬───────┘  └──────────────────────┘
               │
               ↓
        ┌─────────────────────────────┐
        │ PREDICTIONS OUTPUT          │
        │ 1,000 predictions + scores  │
        │ Ready for business use      │
        └─────────────────────────────┘
```

**Key Principles:**
- **Data lineage:** Track where data comes from at each step
- **Reproducibility:** Same preprocessing for training and inference
- **Feature engineering:** Applied consistently everywhere
- **Monitoring:** Track model performance over time

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "File not found" when running scripts

**Solution:**
Ensure you're in the `module1` directory:
```bash
cd module1
python 01_ingest.py
```

#### Issue: Script takes longer than expected or hangs

**Solution:**
Check that required packages are installed:
```bash
pip install pandas scikit-learn mlflow numpy
```

For training scripts specifically, this can be resource-intensive on slower systems. You can skip training and use pre-trained models if needed.

#### Issue: "MLflow experiment not found"

**Solution:**
Make sure you ran the training script first. Training creates the experiment and model artifacts.

#### Issue: Model deployment fails

**Solution:**
1. Verify the training script completed successfully
2. Check that the best model was logged to MLflow
3. Ensure you have proper credentials for Cloudera AI access

#### Issue: Inference script fails with "module not found"

**Solution:**
The inference scripts need the helpers module:
```bash
# Make sure you're in module1 directory
python 05_1_inference_data_prep.py
```

#### Issue: Jupyter notebook won't open

**Solution:**
Make sure you're using Cloudera AI's Jupyter interface. In your project, look for the "Open Workbench" or "Jupyter" button.

### Getting Help

If you encounter issues:
1. Check the error message carefully - it usually indicates the problem
2. Review the step-by-step guide for that section
3. Verify data files exist in the expected locations
4. Check the code comments in the script file
5. Ask your instructor or refer to Cloudera AI documentation

---

## Summary: What You've Accomplished

By completing this lab, you've built a production-grade ML pipeline:

✅ **Data Ingestion** - Loaded data into the data lake
✅ **Data Analysis** - Explored patterns and insights
✅ **Model Training** - Built and tracked multiple experiments
✅ **Model Deployment** - Deployed to Cloudera AI
✅ **Inference Preparation** - Built data transformation pipeline
✅ **Inference Pipeline** - Generated predictions at scale
✅ **Interactive Exploration** - Understood the prediction process

This is how real-world ML systems work. You've seen every step from raw data to production predictions.

---

## Next Steps

- **Advanced Training:** Use `03_train_extended.py` for more model types
- **Custom Features:** Modify feature engineering in the preprocessing module
- **Hyperparameter Tuning:** Adjust model parameters in training scripts
- **Model Monitoring:** Explore Cloudera AI's model monitoring dashboard
- **A/B Testing:** Deploy multiple models and compare performance

---

**Happy Learning!** 🚀

For more information, visit the [Cloudera AI Documentation](https://docs.cloudera.com/cml/).
