import optuna

# --- CONFIGURATION ---
DB_PATH = "sqlite:////Users/benjoel/fypCode/scripts/03_optimisation/optuna_study.db"


def audit_database():
    try:
        summaries = optuna.get_all_study_summaries(storage=DB_PATH)
        if not summaries:
            print("Database is empty — re-run 03_optimised_architecture.py to populate it.")
        else:
            print(f"{len(summaries)} study/studies found:")
            for s in summaries:
                print(f"  Name: '{s.study_name}' — Trials: {s.n_trials}")
    except Exception as e:
        print(f"Error accessing database: {e}")


if __name__ == "__main__":
    audit_database()