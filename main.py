import subprocess

print("\n>>> Running preprocessing (included in pipelines)...")
print("\n>>> Running TF-IDF + XGBoost pipeline...")
subprocess.run(["python", "run_tfidf_pipeline.py"], check=True)
print("\n>>> Running BERT pipeline...")
subprocess.run(["python", "run_bert_pipeline.py"], check=True)
print("\n>>> Analyzing and visualizing results...")
subprocess.run(["python", "analyze.py"], check=True)
print("\n>>> All tasks completed successfully.")