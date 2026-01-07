import subprocess

# Install required packages from requirements.txt
def install_requirements():
    try:
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        print("All packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while installing packages.", e)

if __name__ == "__main__":
    install_requirements()
