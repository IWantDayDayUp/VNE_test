import os
import sys

def main():

    # current_path = os.path.dirname(os.path.realpath(__file__))
    # print(current_path)
    # PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
    # print(PROJECT_HOME)
    # if PROJECT_HOME not in sys.path:
    #     sys.path.append(PROJECT_HOME)
    # print(sys.path)
    
    print(os.path.abspath(os.path.dirname(__file__)))
    print(os.path.basenam(__file__))
    



if __name__ == "__main__":
    main()