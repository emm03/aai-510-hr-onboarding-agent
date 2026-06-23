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
  department_summary.csv

02_hr_onboarding_agent.ipynb
hr_agent.py
five_trace_examples.json
Trace 1.png
Trace 2.png
Trace 3.png
Trace 4.png
Trace 5.png
Agent Evaluation Results and MLflow Trace Analysis
README.md
```

## Data Engineer Notebook

The `01_data_pipeline_eda.ipynb` notebook loads the Employee Onboarding Effectiveness dataset, performs exploratory data analysis, cleans missing values and mixed date formats, creates agent-ready features, and saves cleaned outputs as CSV files and Databricks tables.

Key data engineering steps include:

- Loading the onboarding CSV into Databricks
- Reviewing rows, columns, data types, missing values, and duplicate IDs
- Cleaning mixed date formats
- Filling missing score values using median imputation
- Creating an anonymized `employee_key`
- Creating features such as `days_to_survey`, `overall_onboarding_score`, `low_score_count`, and `onboarding_risk_category`
- Creating department, location, and risk summary tables for downstream agent use

## AI Engineer Notebook

The `02_hr_onboarding_agent.ipynb` notebook presents the agent implementation and evaluation workflow. The agent uses structured onboarding data and tool-based workflows to answer HR questions, summarize risk patterns, and generate data-grounded recommendations.

The AI engineering portion includes:

- Agent prompt and tool workflow
- HR onboarding analysis tools
- Example user questions and responses
- MLflow trace evidence
- Model comparison and evaluation examples
- Graceful rejection examples for out-of-scope requests

## Evaluation Artifacts

The repository includes trace screenshots and evaluation files used to support the final project presentation.

Files include:

- `Trace 1.png`
- `Trace 2.png`
- `Trace 3.png`
- `Trace 4.png`
- `Trace 5.png`
- `five_trace_examples.json`
- `Agent Evaluation Results and MLflow Trace Analysis`

These artifacts show example agent runs, tool usage, response quality, and evaluation evidence.

## Business Value

The HR Onboarding Support Agent reduces manual spreadsheet review by helping HR users ask natural-language questions about onboarding data. The agent can help identify higher-risk employees, departments with weaker onboarding outcomes, and locations that may need additional support.

Potential business value includes:

- Faster HR analysis
- Earlier identification of onboarding risk
- More consistent HR decision support
- Better prioritization of new-hire support resources
- Stronger governance through traceable tool use and evaluation artifacts

## Final Deliverables

This repository supports the final team presentation and includes the notebooks, agent code, trace examples, and evaluation artifacts needed to demonstrate the technical solution and business value of the HR Onboarding Support Agent.
