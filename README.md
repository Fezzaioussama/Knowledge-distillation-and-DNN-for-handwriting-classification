# Knowledge Distillation & DNN for Handwriting Classification

> Two PyTorch experiments — a convolutional network for MNIST digit classification, and knowledge distillation from a larger teacher network into a smaller student.

## The scripts

Both files are plain Python scripts despite having no `.py` extension.

| File | Contents |
|---|---|
| [`DNN for classification of handwriting Digit using Dataset MINSIT`](DNN%20for%20classification%20of%20handwriting%20Digit%20using%20Dataset%20MINSIT) | A `ConvNET` trained on MNIST — convolutional layers, pooling, fully-connected head, training loop, and accuracy plots |
| [`knowledge dislltiled models using Pytorch`](knowledge%20dislltiled%20models%20using%20Pytorch) | `TeacherModel` (larger CNN) and a smaller student trained to match its outputs |

```bash
pip install torch torchvision matplotlib
python "DNN for classification of handwriting Digit using Dataset MINSIT"
```

`torchvision.datasets` downloads MNIST automatically on first run.

## Knowledge distillation

The idea is that a trained network's *full output distribution* carries more
information than the label it predicts. When a teacher classifies a 7, it also
assigns some probability to 1 and a little to 9 — that relative structure
encodes which digits actually resemble each other. A one-hot label throws all of
it away.

Distillation trains the student against the teacher's softened probabilities
(logits divided by a temperature `T` before softmax, which flattens the
distribution and exposes the small values), usually mixed with the ordinary
loss against the true labels. The student ends up better than the same
architecture trained on hard labels alone — the teacher's "dark knowledge" acts
as a richer training signal.

The practical payoff is compression: you get most of a large model's accuracy at
a small model's inference cost, which is why distillation shows up constantly in
edge deployment.

## Related

Pruning, quantization, NAS, and binarized networks are covered in
[Neural-network-optimization-for-edge-device](https://github.com/Fezzaioussama/Neural-network-optimization-for-edge-device),
which includes a second distillation experiment using a ResNet18 teacher.

## Status

Coursework-level experiment scripts — runnable, but not packaged or
parameterised. Hyperparameters and the distillation temperature are inline
constants.
