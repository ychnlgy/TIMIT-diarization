import datetime

FMT = "[%s] %s%s"

class Log:

    def __init__(self, fname):
        self.fname = fname

    def write(self, msg, end="\n"):
        with open(self.fname, "a") as f:
            f.write(
                FMT % (
                    datetime.datetime.now(),
                    msg,
                    end
                )
            )
