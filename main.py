import argparse
from run.run import run

if __name__ == "__main__":
    print("Running the system..")
    parser = argparse.ArgumentParser(description='Running the system')
    parser.add_argument("--do_parse", type=int, required=True)
    parser.add_argument("--runner_configs_path", type=str)
    parser.add_argument("--do_save", type=int)

    arguments = parser.parse_args()

    if arguments.do_parse:
        run(arguments)
    else:
        print("Please run this system through the terminal with required configuration file")
