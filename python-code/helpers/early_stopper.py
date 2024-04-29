import math
import torch
import tempfile
import os
import numpy as np
import logging


# The EarlyStopper is a good measure against overfitting. The moment the train loss keeps decreasing, but the
# validation loss stops decreasing (or even increases), this shows that there's overfitting happening. Having
# a check on the validation loss stops training when this happens

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0, min_epochs=10, logging_level=logging.DEBUG, logger=logging.getLogger('stop')):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.tempdir = tempfile.TemporaryDirectory()
        self.saved_checkpoint_epochs = []  # epochs of saved checkpoints
        self.logger = logger
        self.logger.setLevel(logging_level)
        self.logger.info("Saving checkpoints at %s" % self.tempdir.name)


    def early_stop(self, net, losses, sample_frequency=1, validation_frequency=10):
        # dim 1: loss samples
        def loss_to_epoch(sample_index):
            # indexes and epochs are 0-based
            return (sample_index) * sample_frequency
        def epoch_to_loss(epoch):
            return math.floor(epoch / sample_frequency)
        # dim 2: epochs
        # dim 3: checkpoints
        def epochs_with_checkpoint(epochs): set(epochs) & set(self.saved_checkpoint_epochs)

        # patience is in epochs, patience_samples is length of window in samples
        patience_samples = math.floor(self.patience / sample_frequency)
        current_index = len(losses)-1
        current_epoch = loss_to_epoch( current_index )
        #print("patience length in validation epochs: ", patience_epochs)
        self.logger.debug("current epoch: %d; current index: %d" % (current_epoch, current_index) )
        if current_index < patience_samples:
            # not enough to fill a window yet
            if current_epoch >= self.min_epochs:
                # but more than minimal number of epochs
                self.save_checkpoint(net.model, current_epoch)
            return None
        reference_value = losses[-patience_samples] # start of window
        minimal_value = min(losses[-patience_samples:])
        #print("loss history length:",len(losses))
        #print("loss history window length:",len(losses[-patience_samples:]))
        self.logger.debug("reference value: %.6f" % reference_value)
        self.logger.debug("minimal value: %.6f" % minimal_value)

        # loss, epoch
        epoch_losses = [ (loss, loss_to_epoch(sample_index)) for sample_index, loss in enumerate(losses) ]
        validation_losses = [ (loss, epoch) for loss, epoch in epoch_losses if epoch % validation_frequency == 0 ]
        #print("validation_losses:",validation_losses)
        #print("Min:",max([self.min_epochs, current_epoch-self.patience]))
        window_losses = [ (loss, epoch) for loss,epoch in validation_losses if epoch >= max([self.min_epochs, current_epoch-self.patience]) ]
        #print("window_losses:",window_losses)
        ## Calculate index of best value
        best_in_window = min(window_losses, default=(0,0) )
        self.logger.debug("best in window: %s" % str(best_in_window))
        #best_epoch = loss_to_epoch(best_in_window[1])
        #print("best epoch:",best_epoch)
        # save if best_index == current_index
        if best_in_window[1] == current_epoch:
            self.logger.debug("Saving new best epoch: %d" % current_epoch)
            self.save_checkpoint(net.model, current_epoch)

        if minimal_value < (reference_value-self.min_delta):
            return None # needs to have improved at least min_delta since "patience" validation epochs/losses ago
        if best_in_window[1] < self.min_epochs:
            self.logger.info("best value, but best_epoch %d < min_epochs %d" % (best_in_window[1], self.min_epochs))
            return None

        self.logger.info("stopping! epoch: %d" % current_epoch)
        return current_epoch

    def best_checkpoint_in_window(self, losses, sample_frequency, last_epoch, last_loss):
        def loss_to_epoch(sample_index):
            # indexes are zero-based; after 10 epochs the last index is 9 / after 1 epoch the last index is 0
            return (sample_index) * sample_frequency
        def epoch_to_loss(epoch):
            return math.floor(epoch / sample_frequency)
        self.logger.debug("saved checkpoints: %s" % str(self.saved_checkpoint_epochs))

        # first step, get list of (loss value, index) for those indexes for which we have a checkpoint
        checkpoints_with_losses = [ (losses[epoch_to_loss(epoch)], epoch) for epoch in self.saved_checkpoint_epochs if epoch>= last_epoch-self.patience]

        # second, include last lost-value + index for the last epoch
        checkpoints_with_losses.append( (last_loss, last_epoch) )
        self.logger.debug("checkpoints_with_losses: %s" % str(checkpoints_with_losses))

        # third, get min
        best_in_window = min(checkpoints_with_losses, default=(last_loss, last_epoch))
        self.logger.info("best epoch in window with checkpoint: %s" % str(best_in_window))

        #if len(losses_with_checkpoints_in_window)==0: return None
        return best_in_window[1]

    def save_checkpoint(self, model, epoch):
        '''Saves model when validation loss increases.'''
        filename = self.tempdir.name + "/checkpoint_" + str(epoch)
        self.logger.info("Saving checkpoint to %s" % filename)
        torch.save(model.state_dict(), filename)
        self.saved_checkpoint_epochs.append(epoch)

    def remove_checkpoints(self):
        self.logger.debug("Removing checkpoints from %s" % self.tempdir.name)
        self.tempdir.cleanup()

    def load_checkpoint(self, net, epoch):
        '''Saves model when validation loss increases.'''
        filename = self.tempdir.name + "/checkpoint_" + str(epoch)
        self.logger.info("Loading checkpoint from %s" % filename)
        net.model.load_state_dict(torch.load(filename))



