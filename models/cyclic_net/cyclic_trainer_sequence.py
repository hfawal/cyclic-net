from typing import Dict

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from models.cyclic_net.cyclic_net_sequence import CyclicNetSequence


class CyclicTrainerSequence:

    def __init__(self,
                 model: CyclicNetSequence,
                 criterion: Module,
                 optimizer: str,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 init_lr: Dict[int, float],
                 readout_init_lr: float,
                 thresholds: Dict[int, float]
                 ):
        """
        Initializes a trainer for a generic Cyclic Neural Network given the model, data loaders,
        and hyperparameters.

        :param model: The cyclic net to be trained.
        :param criterion: The loss function to be used on the readout neuron. Non-readout neurons
        use Hinton's goodness-based loss, where goodness is the sum of squared activations.
        :param optimizer: The name of the optimizer to be used for all neurons. Must match
        something from the torch.optim package; this class initilizes optimizers itself.
        :param train_loader: A dataloader that provides contrastive training examples. (See for
        instance the ContrastiveDataset class in utils.)
        :param val_loader: A dataloader that provides contrastive validation examples. (See for
        instance the ContrastiveDataset class in utils.)
        :param test_loader: A dataloader that provides contrastive test examples. (See for
        instance the ContrastiveDataset class in utils.)
        :param device: The device to train on (CPU or GPU).
        :param init_lr: A dictionary of the initial learning rates for each non-readout neuron.
        The keys are neuron IDs, and the values are the learning rates.
        :param readout_init_lr: The initial learning rate for the readout neuron.
        :param thresholds: A dictionary of the threshold hyperparameters for non-readout neurons.
        The keys are neuron IDs, and the values are threshold hyperparameters.
        """

        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.thresholds = thresholds

        # Initialize optimizer per computational neuron.
        self.optimizers: Dict[int, torch.optim.Optimizer] = {
            ID: getattr(torch.optim, optimizer)(
                    self.model.neurons[ID].params,
                    lr=init_lr[ID]
                )
            for ID in self.model.neurons.keys()
        }
        # Initialize readout layer neuron's optimizer.
        self.readout_optimizer: torch.optim.Optimizer = getattr(torch.optim, optimizer)(
            self.model.readout_neuron.params,
            lr=readout_init_lr
        )

        # Save average output sizes used to calculate goodness-based loss.
        self.avg_output_sizes: Dict[int, float] = dict()
        for ID, neuron in self.model.neurons.items():
            size = 0
            for dim in neuron.output_dims.values():
                size += torch.tensor(dim).prod()
            self.avg_output_sizes[ID] = size / len(neuron.output_dims)


    def compute_goodness(self,
                         activations_source: Dict[int, Dict[int, Tensor]],
                         goodness_mask: Tensor
                         ) -> Dict[int, Tensor]:
        """
        Computes the goodness of the outputs of non-readout neurons, following Hinton's formula.
        We generalize the formula to the case that neurons can pass different activations to each
        of their outneighbors - compute the mean over the goodness of each activation vector
        per outneighbor. When neurons are only allowed to pass the same activation to all of
        their outneighbors, this reduces to the same formula as in the Cyclic Net paper.

        :param activations_source: The dictionary of neuron activations, organized by source to
        outneighbor neuron. The entry activations_source[B][A] is the activation that neuron B will
        pass to outneighbor neuron A on the next iteration of the cyclic net propagation.

        :return: A dictionary of the goodness of each neuron's activations. The keys are neuron IDs
        and the values are goodness tensors, shape (B,) where B is the batch size.
        """

        goodness = dict()
        for ID, activations in activations_source.items():

            goodness_per_activation = []
            for actvtn in activations.values():
                actvtn = actvtn[goodness_mask]
                dims_to_sum = list(range(1, len(actvtn.shape)))
                goodness_per_activation.append(actvtn.pow(2).sum(dim=dims_to_sum))

            goodness[ID] = torch.stack(goodness_per_activation, dim=0).mean(dim=0)

        return goodness


    def compute_loss(self,
                     pos_goodness: Dict[int, Tensor],
                     neg_goodness: Dict[int, Tensor]
                     ) -> Dict[int, Tensor]:
        """
        Computes the loss of the outputs of non-readout neurons, following the cyclic net formula.

        :param pos_goodness: The dictionary of goodness computed on positive examples, with keys
        being the neuron IDs and the values being tensors of shape (B,) where B is the batch size.
        :param neg_goodness: The dictionary of goodness computed on negative examples, with keys
        being the neuron IDs and the values being tensors of shape (B,) where B is the batch size.

        :return: A dictionary of the loss of each non-readout neuron. The keys are neuron IDs
        and the values are loss tensors of shape (1,).
        """

        loss = dict()
        for ID in self.model.neurons.keys():

            max_abs = 100 * self.thresholds[ID] * self.avg_output_sizes[ID]
            max_clamp = torch.ones(pos_goodness[ID].shape) * max_abs
            max_clamp = max_clamp.to(pos_goodness[ID].device)
            min_clamp = max_clamp * -1

            pos_input = pos_goodness[ID] - self.avg_output_sizes[ID] * self.thresholds[ID]
            neg_input = self.avg_output_sizes[ID] * self.thresholds[ID] - neg_goodness[ID]
            pos_input = pos_input.clamp(min=min_clamp, max=max_clamp)
            neg_input = neg_input.clamp(min=min_clamp, max=max_clamp)

            pos_term = torch.log(torch.sigmoid(pos_input).clamp(min=1e-25))
            neg_term = torch.log(torch.sigmoid(neg_input).clamp(min=1e-25))

            loss_tensor = -pos_term - neg_term
            loss[ID] = loss_tensor.mean()

        return loss


    def train(self, num_epochs, save_interval, save_dir):
        """
        Trains the cyclic net for the given number of epochs. This modifies the model in-place.

        :param num_epochs: The number of iterations over the training set to train the model.

        :returns: A dictionary of training information with the following structure:
          - validation_accuracy: The validation accuracy over epochs as a python list of floats.
          - neuron_loss: A dictionary where the keys are non-readout neuron IDs, and the values
            are the neuron's loss over optimizer steps, as python lists of floats.
          - readout_loss: The readout neuron's loss over optimizer steps, a python list of floats.
        """

        results = {
            "validation_accuracy": [],
            "neuron_loss": {ID: [] for ID in self.model.neurons.keys()},
            "readout_loss": []
        }

        torch.autograd.set_detect_anomaly(True)

        # Move model to device.
        self.model.to(self.device)

        # Main training loop.
        for epoch in range(1, num_epochs + 1):

            # Batch loop.
            for positive, negative, neutral, label, last_indices in self.train_loader:

                # Move batch data to device.
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                neutral = neutral.to(self.device)
                label = label.to(self.device)
                last_indices = last_indices.to(self.device)
                # Keep activation dictionaries for the positive, negative and neutral.
                pos_act_sink = None
                pos_next_act_source = None
                pos_next_act_sink = None
                neg_act_sink = None
                neg_next_act_source = None
                neg_next_act_sink = None

                N, L, D = positive.shape

                # Propagate the neurons for some number of iterations.
                for i in range(L): # L

                    # Zero parameter gradients for non-readout neurons.
                    for ID in self.model.neurons.keys():
                        self.optimizers[ID].zero_grad()

                    # Propagate the positive inputs.
                    pos_next_act_source, pos_next_act_sink = self.model.propagate(
                        positive[:,i,:],  # Pass input data.
                        pos_act_sink,  # Use previous activations.
                        pos_next_act_source,  # Source-based output dictionary.
                        pos_next_act_sink  # Sink-based output dictionary.
                    )
                    # Swap the dictionaries for the next iteration.
                    pos_act_sink, pos_next_act_sink = pos_next_act_sink, pos_act_sink

                    # Propagate the negative inputs.
                    neg_next_act_source, neg_next_act_sink = self.model.propagate(
                        negative[:,i,:],  # Pass input data.
                        neg_act_sink,  # Use previous activations.
                        neg_next_act_source,  # Source-based output dictionary.
                        neg_next_act_sink  # Sink-based output dictionary.
                    )
                    # Swap the dictionaries for the next iteration.
                    neg_act_sink, neg_next_act_sink = neg_next_act_sink, neg_act_sink

                    goodness_mask = (i<=last_indices)

                    # Compute the goodness and loss for computational neurons.
                    pos_goodness = self.compute_goodness(pos_next_act_source, goodness_mask)
                    neg_goodness = self.compute_goodness(neg_next_act_source, goodness_mask)
                    loss = self.compute_loss(pos_goodness, neg_goodness)

                    # Optimize the computational neurons.
                    for ID in self.model.neurons.keys():
                        loss[ID].backward()
                        self.optimizers[ID].step()
                        # Record the loss.
                        results["neuron_loss"][ID].append(loss[ID].item())

                # Compute the loss for the readout neuron.
                self.readout_optimizer.zero_grad()
                prediction = self.model.forward(neutral, last_indices)
                readout_loss = self.criterion(prediction, label)
                # Record the readout loss.
                results["readout_loss"].append(readout_loss.item())

                # Optimize the readout neuron.
                readout_loss.backward()
                self.readout_optimizer.step()
                print(f"Epoch [{epoch}/{num_epochs}]: Batch {i+1}/{len(self.train_loader)}: Readout Loss {readout_loss.item():.4f}")

            # Compute and record the validation accuracy.
            validation_acc = self.validate()
            results["validation_accuracy"].append(validation_acc)
            print(f"Epoch [{epoch}/{num_epochs}]: Validation Accuracy {validation_acc:.2f}%")
            if epoch+1 % save_interval == 0:
                torch.save(self.model.state_dict(), f"{save_dir}/model_{epoch}.pth")

        return results

    @torch.no_grad()
    def validate(self) -> float:
        """
        Computes the accuracy of the current model on the validation set. This assumes that the
        purpose of the cyclic net is a classification task.

        :return: The accuracy of the model on the validation set, as a float between 0 and 100.
        """

        # Keep track of totals.
        correct = 0
        total = 0
        with torch.no_grad():

            # Batch loop.
            for positive, negative, neutral, label, last_indices in self.val_loader:

                # Move batch data to device.
                neutral = neutral.to(self.device)
                label = label.to(self.device)
                last_indices = last_indices.to(self.device)

                # Compute the number of correct predictions.
                output = self.model.forward(neutral, last_indices)
                _, predicted = torch.max(output.data, dim=1)
                total += label.shape[0]
                correct += torch.sum(predicted == label).item()

        accuracy = 100 * correct / total
        return accuracy


    def test(self) -> float:
        """
        Computes the accuracy of the current model on the validation set. This assumes that the
        purpose of the cyclic net is a classification task.

        :return: The accuracy of the model on the validation set, as a float between 0 and 100.
        """

        # Keep track of totals.
        correct = 0
        total = 0
        with torch.no_grad():

            # Batch loop.
            for positive, negative, neutral, label, last_indices in self.test_loader:

                # Move batch data to device.
                neutral = neutral.to(self.device)
                label = label.to(self.device)
                last_indices = last_indices.to(self.device)

                # Compute the number of correct predictions.
                output = self.model.forward(neutral, last_indices)
                _, predicted = torch.max(output.data, dim=1)
                total += label.shape[0]
                correct += torch.sum(predicted == label).item()

        accuracy = 100 * correct / total
        return accuracy