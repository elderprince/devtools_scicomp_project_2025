import torchmetrics.classification.accuracy as accuracy
from torch.nn import CrossEntropyLoss
import lightning as L
import torch

class Classifier(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        # Define the model
        self.model = model()
        # Define the loss
        self.loss = CrossEntropyLoss()
        # Define the accuracy
        self.train_accuracy = accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.valid_accuracy = accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.test_accuracy = accuracy(task="multiclass", num_classes=self.model.num_classes)
    
    def _classifier_step(self, batch): 
        # Extract features
        images, labels = batch
        # Compute the logits and predict the label
        y_pred = self(images)
        preds = torch.argmax(y_pred, 1)
        # Comute crossentropy
        loss = self.loss(preds, labels)
        # Compute
        acc = self.train_accuracy(preds, labels)
        return preds, loss, acc
    
    def training_step(self, batch):
        preds, loss, acc = self._classifier_step(batch)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch): 
        preds, loss, acc = self._classifier_step(batch)
        self.log('valid_accuracy', self.valid_accuracy, on_step=True, on_epoch=False)
        return loss
    
    def test_step(self, batch): 
        preds, loss, acc = self._classifier_step(batch)
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=False)
        return loss
    
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer