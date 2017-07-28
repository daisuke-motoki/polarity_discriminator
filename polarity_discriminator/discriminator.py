import logging
import keras
from keras.models import model_from_json
from polarity_discriminator.models import network_model1
from polarity_discriminator.losses import SumPolarityLoss

logger = logging.getLogger(__name__)


class PolarityDiscriminator:
    """
    """
    def __init__(self):
        """
        """
        self.model = None

    def build(self, architecture):
        """
        """
        # create network
        self.model = network_model1(architecture)

    def save_model(self, filename):
        """
        """
        with open(filename + "_layer.json", "w") as file_:
            file_.write(self.model.to_json(**{"indent": 4}))
        self.model.save_weights(filename + "_weights.hdf5")

    def load_model(self, filename):
        """
        """
        with open(filename + "_layer.json", "r") as file_:
            self.model = model_from_json(file_.read())
        self.model.load_weights(filename + "_weights.hdf5")

    def train(self,
              input_data,
              Y,
              validation_data,
              validation_Y,
              epochs=10,
              batch_size=50,
              learning_rate=1e-3,
              checkpoints=None,
              shuffle=True):
        """
        """
        # loss & optimizer
        loss = SumPolarityLoss()
        optim = keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(
            optimizer=optim,
            loss=loss.compute_loss,
            # metric=["acc"]
        )
        # call backs
        callbacks = list()
        if checkpoints:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    checkpoints,
                    verbose=1,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True
                )
            )

        def schedule(epoch, decay=0.9):
            return learning_rate * decay**(epoch)
        callbacks.append(keras.callbacks.LearningRateScheduler(schedule))

        # fit
        self.model.fit(input_data,
                       Y,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=callbacks,
                       validation_data=[validation_data, validation_Y],
                       shuffle=shuffle)

    def predict(self,
                input_data,
                batch_size=32):
        """
        """
        return self.model.predict(input_data,
                                  batch_size=batch_size)
