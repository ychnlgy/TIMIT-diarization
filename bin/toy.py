import argparse

from __init__ import src

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--email")
    args = parser.parse_args()

    acc = src.training.toy.main()

    if args.email is not None:
        import mailupdater
        service = mailupdater.Service(args.email)

        with service.create("Acc %.2f" % acc):
            pass
