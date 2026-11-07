# EdgeTorch

## 10 epochs with Adam
Epoch 01 | Loss: 1.7995 | Train: 32.5% | Val: 38.0%
Epoch 02 | Loss: 1.3404 | Train: 51.6% | Val: 54.9%
Epoch 03 | Loss: 1.1119 | Train: 59.1% | Val: 57.5%
Epoch 04 | Loss: 0.9549 | Train: 65.3% | Val: 65.2%
Epoch 05 | Loss: 0.8480 | Train: 69.9% | Val: 68.7%
Epoch 06 | Loss: 0.7701 | Train: 72.7% | Val: 70.2%
Epoch 07 | Loss: 0.7082 | Train: 75.0% | Val: 73.1%
Epoch 08 | Loss: 0.6645 | Train: 76.1% | Val: 74.8%
Epoch 09 | Loss: 0.6077 | Train: 78.5% | Val: 73.4%
Epoch 10 | Loss: 0.5689 | Train: 79.5% | Val: 76.3%
Best validation accuracy: 76.3%
Test accuracy: 71.2%

## 30 epochs with Adam
Epoch 01 | Loss: 1.8833 | Train: 29.1% | Val: 38.2%
Epoch 02 | Loss: 1.4333 | Train: 47.7% | Val: 50.4%
Epoch 03 | Loss: 1.1989 | Train: 57.3% | Val: 51.2%
Epoch 04 | Loss: 1.0595 | Train: 60.8% | Val: 59.9%
Epoch 05 | Loss: 0.9795 | Train: 63.9% | Val: 64.4%
Epoch 06 | Loss: 0.8923 | Train: 67.6% | Val: 62.0%
Epoch 07 | Loss: 0.8318 | Train: 70.1% | Val: 69.0%
Epoch 08 | Loss: 0.7820 | Train: 71.2% | Val: 66.1%
Epoch 09 | Loss: 0.7310 | Train: 73.9% | Val: 73.1%
Epoch 10 | Loss: 0.6740 | Train: 75.7% | Val: 71.9%
Epoch 11 | Loss: 0.6249 | Train: 77.7% | Val: 76.6%
Epoch 12 | Loss: 0.5848 | Train: 78.9% | Val: 76.7%
Epoch 13 | Loss: 0.5598 | Train: 79.8% | Val: 77.8%
Epoch 14 | Loss: 0.5247 | Train: 81.8% | Val: 78.7%
Epoch 15 | Loss: 0.4919 | Train: 82.6% | Val: 82.4%
Epoch 16 | Loss: 0.4721 | Train: 83.3% | Val: 82.4%
Epoch 17 | Loss: 0.4446 | Train: 84.5% | Val: 81.0%
Epoch 18 | Loss: 0.4402 | Train: 84.4% | Val: 81.0%
Epoch 19 | Loss: 0.4160 | Train: 85.4% | Val: 83.9%
Epoch 20 | Loss: 0.3795 | Train: 86.8% | Val: 82.2%
Epoch 21 | Loss: 0.3782 | Train: 86.6% | Val: 83.7%
Epoch 22 | Loss: 0.3666 | Train: 87.6% | Val: 80.6%
Epoch 23 | Loss: 0.3647 | Train: 87.5% | Val: 84.6%
Epoch 24 | Loss: 0.3380 | Train: 88.6% | Val: 84.1%
Epoch 25 | Loss: 0.3375 | Train: 88.6% | Val: 82.7%
Epoch 26 | Loss: 0.3350 | Train: 88.3% | Val: 84.3%
Epoch 27 | Loss: 0.3005 | Train: 89.8% | Val: 84.7%
Epoch 28 | Loss: 0.3023 | Train: 89.7% | Val: 84.1%
Epoch 29 | Loss: 0.2948 | Train: 89.9% | Val: 82.5%
Epoch 30 | Loss: 0.2935 | Train: 89.9% | Val: 84.8%
Best validation accuracy: 84.8%
Test accuracy: 80.8%

comment: after epoch 15 it seems that validation stagnates while
training accuracy continues to improve suggesting some overfitting

## 30 epochs with SGD
Epoch 01 | Loss: 1.9810 | Train: 22.1% | Val: 25.5%
Epoch 02 | Loss: 1.6676 | Train: 35.8% | Val: 36.4%
Epoch 03 | Loss: 1.4043 | Train: 46.7% | Val: 45.2%
Epoch 04 | Loss: 1.1800 | Train: 54.4% | Val: 58.6%
Epoch 05 | Loss: 1.0007 | Train: 61.9% | Val: 66.0%
Epoch 06 | Loss: 0.9184 | Train: 66.1% | Val: 65.9%
Epoch 07 | Loss: 0.7999 | Train: 70.6% | Val: 73.6%
Epoch 08 | Loss: 0.6957 | Train: 75.1% | Val: 68.3%
Epoch 09 | Loss: 0.6502 | Train: 76.4% | Val: 76.4%
Epoch 10 | Loss: 0.5373 | Train: 81.0% | Val: 80.9%
Epoch 11 | Loss: 0.4893 | Train: 83.0% | Val: 80.6%
Epoch 12 | Loss: 0.4113 | Train: 85.9% | Val: 78.9%
Epoch 13 | Loss: 0.3810 | Train: 86.6% | Val: 84.2%
Epoch 14 | Loss: 0.3483 | Train: 87.8% | Val: 84.1%
Epoch 15 | Loss: 0.3458 | Train: 87.6% | Val: 84.3%
Epoch 16 | Loss: 0.3099 | Train: 89.0% | Val: 83.6%
Epoch 17 | Loss: 0.2803 | Train: 90.2% | Val: 85.4%
Epoch 18 | Loss: 0.2790 | Train: 89.9% | Val: 86.0%
Epoch 19 | Loss: 0.2423 | Train: 91.3% | Val: 84.7%
Epoch 20 | Loss: 0.2257 | Train: 92.2% | Val: 86.0%
Epoch 21 | Loss: 0.1991 | Train: 92.7% | Val: 86.8%
Epoch 22 | Loss: 0.2031 | Train: 93.2% | Val: 86.8%
Epoch 23 | Loss: 0.1823 | Train: 93.4% | Val: 88.2%
Epoch 24 | Loss: 0.2136 | Train: 92.3% | Val: 85.6%
Epoch 25 | Loss: 0.1744 | Train: 93.8% | Val: 86.8%
Epoch 26 | Loss: 0.1761 | Train: 94.0% | Val: 87.3%
Epoch 27 | Loss: 0.1637 | Train: 94.3% | Val: 88.6%
Epoch 28 | Loss: 0.1428 | Train: 95.1% | Val: 88.0%
Epoch 29 | Loss: 0.1328 | Train: 95.1% | Val: 86.0%
Epoch 30 | Loss: 0.1547 | Train: 94.6% | Val: 88.2%
Best validation accuracy: 88.6%
Test accuracy: 83.9%

comment: It seems that SGD performed better than Adam
possibly due to momentum helping it make consistent progress

## 30 epochs with SGD + 20% dropout
Epoch 01 | Loss: 2.0457 | Train: 18.1% | Val: 23.6%
Epoch 02 | Loss: 1.8581 | Train: 27.2% | Val: 30.0%
Epoch 03 | Loss: 1.6156 | Train: 37.7% | Val: 38.4%
Epoch 04 | Loss: 1.3975 | Train: 45.9% | Val: 48.4%
Epoch 05 | Loss: 1.2868 | Train: 50.4% | Val: 52.3%
Epoch 06 | Loss: 1.1521 | Train: 55.1% | Val: 64.1%
Epoch 07 | Loss: 1.0633 | Train: 59.1% | Val: 63.9%
Epoch 08 | Loss: 1.0127 | Train: 60.8% | Val: 69.1%
Epoch 09 | Loss: 0.9631 | Train: 63.3% | Val: 64.4%
Epoch 10 | Loss: 0.8901 | Train: 66.4% | Val: 71.5%
Epoch 11 | Loss: 0.8084 | Train: 69.9% | Val: 75.0%
Epoch 12 | Loss: 0.7415 | Train: 72.0% | Val: 69.3%
Epoch 13 | Loss: 0.6778 | Train: 74.9% | Val: 80.7%
Epoch 14 | Loss: 0.6279 | Train: 77.0% | Val: 80.7%
Epoch 15 | Loss: 0.5731 | Train: 79.3% | Val: 81.8%
Epoch 16 | Loss: 0.5347 | Train: 80.2% | Val: 82.3%
Epoch 17 | Loss: 0.4969 | Train: 81.8% | Val: 85.5%
Epoch 18 | Loss: 0.4728 | Train: 82.7% | Val: 84.7%
Epoch 19 | Loss: 0.4460 | Train: 84.1% | Val: 83.9%
Epoch 20 | Loss: 0.4126 | Train: 85.1% | Val: 84.8%
Epoch 21 | Loss: 0.3853 | Train: 86.3% | Val: 86.1%
Epoch 22 | Loss: 0.3821 | Train: 86.7% | Val: 86.6%
Epoch 23 | Loss: 0.3507 | Train: 87.2% | Val: 86.7%
Epoch 24 | Loss: 0.3410 | Train: 87.6% | Val: 84.3%
Epoch 25 | Loss: 0.3176 | Train: 88.5% | Val: 87.1%
Epoch 26 | Loss: 0.3217 | Train: 88.5% | Val: 87.2%
Epoch 27 | Loss: 0.2823 | Train: 89.9% | Val: 87.1%
Epoch 28 | Loss: 0.2776 | Train: 90.3% | Val: 88.4%
Epoch 29 | Loss: 0.2668 | Train: 90.3% | Val: 86.7%
Epoch 30 | Loss: 0.2612 | Train: 90.7% | Val: 87.3%
Best validation accuracy: 88.4%
Test accuracy: 85.1%

comment: using dropout as a regularization technique slowed down
our training however it did seem to help with overfitting. But both
models ultimately seemed to generalise similarly as our validation differs
by just 0.2

## Inference Speed Ryzen 5 5600 1 thread
Mean latency: 0.569 ms
Median latency: 0.522 ms
Min latency: 0.417 ms
Max latency: 1.148 ms

## Inference speed with ONNX FP32
Max output difference: 0.00000191

Mean latency: 0.131 ms
Median latency: 0.121 ms
Min latency: 0.100 ms
Max latency: 1.452 ms

comment: with ONNX we can see a huge increase in performance
with the same trained model by using optimized kernels

## Quantisation results
21K models/speech_cnn.onnx
154K models/speech_cnn.onnx.data
70K models/speech_cnn_int8.onnx

comment: INT8 quantisation worked fantastic with a 2.5x reduction in 
size compared to the FP32 model. Though it can be noted that the model
is not exactly 4x smaller due to some operations and metadata
requiring additional storage

## Inference speed with ONNX INT8
(200 sample calibration)
Mean latency: 0.234 ms
Median latency: 0.212 ms
Min latency: 0.187 ms
Max latency: 1.570 ms

(1000 sample calibration)
Mean latency: 0.235 ms
Median latency: 0.214 ms
Min latency: 0.187 ms
Max latency: 0.693 ms

## FP32 vs INT8 (200 sample calibration) ONNX accuracy
ONNX FP32: 88.40%
ONNX INT8: 86.00%
Samples evaluated: 836

### INT8 (1000 sample calibration)
ONNX INT8: 86.60%

## confusion matrix results
FP32 accuracy: 88.40%
down: 84/95 correct
go: 82/109 correct
left: 108/113 correct
no: 93/105 correct
right: 98/110 correct
stop: 86/100 correct
up: 101/109 correct
yes: 87/95 correct

INT8 accuracy: 86.60%
down: 84/95 correct
go: 71/109 correct
left: 106/113 correct
no: 98/105 correct
right: 98/110 correct
stop: 84/100 correct
up: 97/109 correct
yes: 86/95 correct