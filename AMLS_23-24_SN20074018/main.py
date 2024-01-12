import os
import subprocess

def list_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.py')]

def run_script(path):
    subprocess.run(['python', path], check=True)

def main():
    while True:
        print("Select the script to run:")
        print("A - Run scripts in A directory")
        print("B - Run scripts in B directory")
        print("Q - Quit")
        choice = input("Enter your choice: ").upper()

        if choice == 'Q':
            break
        elif choice in ['A', 'B']:
            dir_path = f"C:\\Users\\An\\AMLS_23-24_SN20074018\\{choice}"
            scripts = list_files(dir_path)
            for i, script in enumerate(scripts):
                print(f"{i+1}. {script}")
            script_choice = int(input("Enter the number of the script to run: "))
            if 1 <= script_choice <= len(scripts):
                run_script(os.path.join(dir_path, scripts[script_choice-1]))
            else:
                print("Invalid selection.")
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
