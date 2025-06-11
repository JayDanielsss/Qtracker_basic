import subprocess
import re

#This script is for running the momentum model, saving the results, and creating a residual plot.


def run_training_and_get_ith_model(mup_root, mum_root):
    #This function simply determines what trail of model you are on and runs Momentum_training.py


    cmd = [
        "python3",
        "training_scripts/Momentum_training.py",
        mup_root,
        mum_root
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    full_output = ""
    # Stream output live
    for line in process.stdout:
        print(line, end="")  # Print live output (no extra newline)
        full_output += line

    process.wait()

    if process.returncode != 0:
        print(f"Training script failed with return code {process.returncode}")
        return None

    # Extract ith model from full output after training
    match = re.search(r"Next model number: (\d+)", full_output)
    if match:
        ith_model = int(match.group(1))
        return ith_model
    else:
        print("Could not find model index in output.")
        return None
    

def run_qtracker_prod(root_file):

    #This runs the final program in the pipeline, qtrack_prod. This will produce a root file that will
    #be sent to the evaluator code. 
    cmd = [
        "python3",
        "QTracker_prod.py",
        root_file
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    print(f"\n--- Running QTracker_prod.py on {root_file} ---\n")
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"QTracker_prod.py failed with return code {process.returncode}")
    else:
        print("\nQTracker_prod.py completed successfully.")

def run_momentum_res(output_root, ith_model):
    cmd = [
        "python3",
        "Util/Momentum_res.py",
        output_root,
        str(ith_model)
    ]

    print(f"\n--- Running Momentum_res.py with {output_root} and ith_Model={ith_model} ---\n")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"Momentum_res.py failed with return code {process.returncode}")
    else:
        print("\nMomentum_res.py completed successfully.")

if __name__ == "__main__":

    #These files are from running the setup for training, gen_training.py.
    mup_file = "momentum_training-1.root"
    mum_file = "momentum_training-2.root"
    #Data file.
    root_file = "data/DY_Target_10K.root"
    #Output from Qtracker.
    output_root = "qtracker_reco.root"

    model_index = run_training_and_get_ith_model(mup_file, mum_file)


    if model_index is not None:
        print(f"Training complete. Trained model index: {model_index}")
        run_qtracker_prod(root_file)
        run_momentum_res(output_root, model_index)



#Future work...
#Sanity check, how does the error going into the model effects things. Add noise/ smearing to the track.
#Add width and position
#Use J/Psi. 