class RandomInit:

    def __init__(self, random_init=True, always=True, mod=500, start=100, end=300):
        self.random_initialization = random_init
        self.always_random = always
        self.mod = int(mod) if mod else False
        self.start = int(start) if start else False
        self.end = int(end) if end else False
        if self.random_initialization and not self.always_random:
            assert type(self.mod) == int, "mod is not a number"
            assert type(self.start) == int, "start is not a number"
            assert type(self.end) == int, "end is not a number"

    def use_random_init(self, episode):
        if not self.random_initialization:
            return False
        if self.always_random:
            return True
        if self.start <= episode % self.mod < self.end:
            return True
        return False
