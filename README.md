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
  02_hr_onboarding_agent.ipynb
```

## Data Engineer and AI Engineer Notebooks

- The `01_data_pipeline_eda.ipynb` notebook loads the Employee Onboarding Effectiveness dataset, performs EDA, cleans missing values and mixed date formats, creates agent-ready features, and saves cleaned outputs as CSV files and Databricks tables.

- The `02_hr_onboarding_agent.ipynb` notebook presents the Artificial Intelligence Engineer portion of the final team project. The goal is to build and evaluate an HR Onboarding Insights Agent that helps HR managers analyze onboarding survey data, identify departments and locations with weaker onboarding outcomes, summarize risk patterns related to probation attrition, and generate data-grounded recommendations.
