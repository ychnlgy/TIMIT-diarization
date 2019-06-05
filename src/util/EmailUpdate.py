import time, os

class EmailUpdate:

    def __init__(self, target_path, sleep=2):
        self.fname = target_path
        self.sleep = sleep
        self.last = 0
        self.counter = 0
        self.service = None

    def main(self, username):

        import mailupdater

        self.service = mailupdater.Service(username)

        while True:
            if self.check_modified():
                self.counter += 1
                self.send()
            time.sleep(self.sleep)

    # === PROTECTED ===

    def check_modified(self):
        if not os.path.isfile(self.fname):
            return False
        else:
            modt = os.path.getmtime(self.fname)
            if modt > self.last + self.sleep:
                self.last = modt
                return True
            else:
                return False

    def send(self):
        with self.service.create("Update %d for %s" % (self.counter, self.fname)) as email:
            email.attach(self.fname)
