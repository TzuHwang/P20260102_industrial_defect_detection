class Loss_Funcs:
    def __init__(self, args):
        self.loss_fcns = {}
        self.loss_values = {}
        self.in_channel_loss_values = {}
        self.losses = args.losses
        self.channel_weights = args.channel_weights
        self.weights = args.loss_weights

    def get_loss_fcns(self):
        return self.loss_fcns

    def get_loss_value(self, pred, target):
        pass
