import argparse

from __init__ import src

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--email")
    args = parser.parse()
    
    src.util.EmailUpdate("results.txt").main(args.email)
