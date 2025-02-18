import subprocess
import sys


def run_script(script_path):
    print(f"Running {script_path}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running {script_path}:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        print(f"{script_path} completed successfully.\n")


def main():
    """
    PIPELINE STEPS:
    1. Split images into patches
    2. Run inference on patches
    3. Reconstruct images
    """
    scripts = [
        "src/processing/mass_split.py",  # Step 1: Split images into patches
        "src/processing/batch_inference.py",  # Step 2: Run inference on patches
        "src/processing/mass_reconstruct.py",  # Step 3: Reconstruct images
    ]

    for script in scripts:
        run_script(script)

    print("All steps completed successfully.")


if __name__ == "__main__":
    main()
