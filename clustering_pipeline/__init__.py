"""
Semantic Hierarchical Clustering Pipeline for AWS Glue 5.0 + PySpark.

Two jobs:
    - job_training:   Trains 3-level KMeans models and names clusters via LLM.
    - job_assignment: Daily job that embeds new phrases and assigns cluster labels.
"""
