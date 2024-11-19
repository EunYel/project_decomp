import argparse
from multiprocessing import Process
from pathlib import Path
from preprocess.preprocess_brics import process_smiles as process_brics
from preprocess.preprocess_recap import process_smiles as process_recap
from preprocess.preprocess_jt import process_smiles as process_jt

def run_process(function, input_file, task_name):
    try:
        print(f"Starting {task_name} preprocessing...")
        function(input_file)
        print(f"{task_name} preprocessing completed successfully.")
    except Exception as e:
        print(f"Error during {task_name} preprocessing: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing scripts for RECAP, BRICS, and JT decomposition in parallel.")
    parser.add_argument("input_file", help="Path to the input CSV file containing SMILES strings.")
    args = parser.parse_args()

    input_file = args.input_file
    if not Path(input_file).exists():
        print(f"Error: Input file does not exist: {input_file}")
        return

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Define processes
    processes = [
        Process(target=run_process, args=(process_brics, input_file, "BRICS")),
        Process(target=run_process, args=(process_recap, input_file, "RECAP")),
        Process(target=run_process, args=(process_jt, input_file, "JT"))
    ]

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("All preprocessing tasks completed successfully.")

if __name__ == "__main__":
    main()
