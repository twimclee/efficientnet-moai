class MHyp:

    def __init__(self):
        self.model = None
        self.epoch = None
        self.batch_size = None
        self.num_class = None
        self.learning_rate = None
        self.weight_decay = None
        self.img_size = None
        self.optim = None
        self.gpu = None
        self.loss = None
        self.lr_lambda = None

        self.brightness = None
        self.contrast = None
        self.saturation = None
        self.hue = None
        self.hflip = None
        self.vflip = None
        self.rotate = None

    def print_data(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

class MData:

    def __init__(self):
        self.names = None

    def print_data(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

