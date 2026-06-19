# AAI-510 HR Onboarding Support Agent

Final team project for AAI-510: Agentic AI Systems.

This project builds an HR Onboarding Support Agent that analyzes onboarding survey data and provides HR decision-support insights. The agent is intended to help HR teams identify onboarding risks, summarize department and location trends, and generate recommendations for improving new-hire support.

## Team Members

- Emmi Bishop - Data Engineer
- Peng Wang - AI Engineer
- Glen Salazar - Product Manager

## Project Structure

```text
notebooks/
  01_data_pipeline_eda.ipynb
```

## Data Engineer Notebook

The `01_data_pipeline_eda.ipynb` notebook loads the Employee Onboarding Effectiveness dataset, performs EDA, cleans missing values and mixed date formats, creates agent-ready features, and saves cleaned outputs as CSV files and Databricks tables.
