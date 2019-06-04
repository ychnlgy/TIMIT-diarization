import argparse

from __init__ import src

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--email")
    args = parser.parse_args()

    service = None
    if args.email is not None:
        import mailupdater
        service = mailupdater.Service(args.email)

    acc = src.training.toy.main()

    if service is not None:
        with service.create("Acc %.2f" % acc):
            pass
