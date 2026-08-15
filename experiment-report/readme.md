final results:

First truncation results:
truncation_results.png

GloVe coverage: 18956/20000 words (95%) missing words keep random init
embedding matrix: (20000, 100)

LSTM Basline results

LSTM trainable params: 2118018
[LSTM] epoch  1/15 | train loss 0.568 acc 0.720 | val loss 0.583 acc 0.698
[LSTM] epoch  2/15 | train loss 0.463 acc 0.780 | val loss 0.510 acc 0.747
[LSTM] epoch  3/15 | train loss 0.314 acc 0.873 | val loss 0.441 acc 0.807
[LSTM] epoch  4/15 | train loss 0.220 acc 0.922 | val loss 0.412 acc 0.839
[LSTM] epoch  5/15 | train loss 0.148 acc 0.952 | val loss 0.412 acc 0.828
[LSTM] epoch  6/15 | train loss 0.181 acc 0.933 | val loss 0.503 acc 0.798
[LSTM] epoch  7/15 | train loss 0.055 acc 0.986 | val loss 0.466 acc 0.828
  early stop at epoch 7 (no val gain for 3 epochs)
  restored best val acc 0.839 (epoch 4)

Baseline LSTM test accuracy: 0.824

image/validation_loss_and_validaton_accuracy.png which has the validation loss and validation accuracy 

Validation eval

OK  pred POS (p=0.91) | true POS | This movie was awesome. It's a documentary about how surfing influenced skateboarding in the early days. It has interviews with skaters such as Tony Hawk(my ido...
OK  pred POS (p=0.95) | true POS | When the Italians and Miles O'keeffe work together nothing can go wrong! As ever, Miles is great as the almost as great Ator; the most lovable barbarian of all ...
OK  pred NEG (p=0.96) | true NEG | Forget what I said about Emeril. Rachael Ray is the most irritating personality on the Food Network AND all of television. If you've never seen 30 Minute Meals,...
OK  pred POS (p=0.91) | true POS | This is one of my 3 favorite movies. I've been out on the water since I was 13, so I got a lot of the humor as well as recognizing a lot of the near-land scener...
XX  pred POS (p=0.97) | true NEG | Oh boy ! It was just a dream ! What a great idea ! Mr Lynch is very lucky most people try to tell classical stories. This way he can play with his little planti...
OK  pred POS (p=0.92) | true POS | This is one of my favorite T.V shows of all time, Rowan Atkinson is simply a genius!, and it's only fitting that i chose this to be my 1000 review!. I can't beg...

Model zoo comparison

RNN   | test acc 0.755 | best val 0.779 @ep 5 | params 2,029,698 | 19.6s
LSTM  | test acc 0.824 | best val 0.837 @ep 7 | params 2,118,018 | 24.6s
GRU   | test acc 0.839 | best val 0.866 @ep 6 | params 2,088,578 | 21.1s
CNN   | test acc 0.854 | best val 0.865 @ep 4 | params 2,120,902 | 11.2s


## rnn vs lstm experiment (try 1)

filler T=  5 | RNN acc 1.000 | LSTM acc 1.000
filler T= 10 | RNN acc 1.000 | LSTM acc 1.000
filler T= 20 | RNN acc 1.000 | LSTM acc 1.000
filler T= 35 | RNN acc 1.000 | LSTM acc 1.000
filler T= 50 | RNN acc 0.530 | LSTM acc 0.526

## section 8

[OVERFIT] epoch  1/25 | train loss 0.684 acc 0.645 | val loss 0.690 acc 0.554
[OVERFIT] epoch  2/25 | train loss 0.669 acc 0.647 | val loss 0.686 acc 0.566
[OVERFIT] epoch  3/25 | train loss 0.650 acc 0.695 | val loss 0.860 acc 0.614
[OVERFIT] epoch  4/25 | train loss 0.493 acc 0.777 | val loss 0.674 acc 0.603
[OVERFIT] epoch  5/25 | train loss 0.581 acc 0.730 | val loss 1.006 acc 0.565
[OVERFIT] epoch  6/25 | train loss 0.356 acc 0.868 | val loss 0.722 acc 0.590
[OVERFIT] epoch  7/25 | train loss 0.231 acc 0.910 | val loss 0.940 acc 0.589
[OVERFIT] epoch  8/25 | train loss 0.151 acc 0.953 | val loss 1.075 acc 0.617
[OVERFIT] epoch  9/25 | train loss 0.154 acc 0.927 | val loss 1.570 acc 0.606
[OVERFIT] epoch 10/25 | train loss 0.036 acc 0.995 | val loss 1.285 acc 0.603
[OVERFIT] epoch 11/25 | train loss 0.022 acc 0.995 | val loss 1.458 acc 0.631
[OVERFIT] epoch 12/25 | train loss 0.008 acc 1.000 | val loss 1.634 acc 0.619
[OVERFIT] epoch 13/25 | train loss 0.009 acc 0.995 | val loss 2.132 acc 0.623
[OVERFIT] epoch 14/25 | train loss 0.003 acc 1.000 | val loss 1.783 acc 0.613
[OVERFIT] epoch 15/25 | train loss 0.008 acc 0.998 | val loss 1.818 acc 0.599
[OVERFIT] epoch 16/25 | train loss 0.012 acc 0.998 | val loss 2.424 acc 0.605
[OVERFIT] epoch 17/25 | train loss 0.007 acc 1.000 | val loss 1.594 acc 0.617
[OVERFIT] epoch 18/25 | train loss 0.011 acc 1.000 | val loss 2.062 acc 0.609
[OVERFIT] epoch 19/25 | train loss 0.001 acc 1.000 | val loss 1.941 acc 0.631
[OVERFIT] epoch 20/25 | train loss 0.001 acc 1.000 | val loss 2.133 acc 0.626
[OVERFIT] epoch 21/25 | train loss 0.000 acc 1.000 | val loss 2.134 acc 0.631
[OVERFIT] epoch 22/25 | train loss 0.000 acc 1.000 | val loss 2.202 acc 0.629
[OVERFIT] epoch 23/25 | train loss 0.000 acc 1.000 | val loss 2.288 acc 0.629
[OVERFIT] epoch 24/25 | train loss 0.000 acc 1.000 | val loss 2.336 acc 0.628
[OVERFIT] epoch 25/25 | train loss 0.000 acc 1.000 | val loss 2.365 acc 0.624
  restored best val acc 0.631 (epoch 19)


images/8a_overfitting.png

[REGULARIZED] epoch  1/25 | train loss 0.688 acc 0.557 | val loss 0.691 acc 0.546
[REGULARIZED] epoch  2/25 | train loss 0.683 acc 0.562 | val loss 0.689 acc 0.533
[REGULARIZED] epoch  3/25 | train loss 0.673 acc 0.650 | val loss 0.684 acc 0.566
[REGULARIZED] epoch  4/25 | train loss 0.665 acc 0.580 | val loss 0.694 acc 0.565
[REGULARIZED] epoch  5/25 | train loss 0.653 acc 0.593 | val loss 0.690 acc 0.567
[REGULARIZED] epoch  6/25 | train loss 0.628 acc 0.670 | val loss 0.671 acc 0.570
[REGULARIZED] epoch  7/25 | train loss 0.593 acc 0.677 | val loss 0.690 acc 0.618
[REGULARIZED] epoch  8/25 | train loss 0.608 acc 0.670 | val loss 0.680 acc 0.561
[REGULARIZED] epoch  9/25 | train loss 0.583 acc 0.690 | val loss 0.681 acc 0.583
[REGULARIZED] epoch 10/25 | train loss 0.524 acc 0.715 | val loss 0.679 acc 0.629
[REGULARIZED] epoch 11/25 | train loss 0.537 acc 0.725 | val loss 0.676 acc 0.601
[REGULARIZED] epoch 12/25 | train loss 0.506 acc 0.748 | val loss 0.702 acc 0.609
[REGULARIZED] epoch 13/25 | train loss 0.439 acc 0.805 | val loss 0.720 acc 0.613
[REGULARIZED] epoch 14/25 | train loss 0.416 acc 0.818 | val loss 0.705 acc 0.625
  early stop at epoch 14 (no val gain for 4 epochs)
  restored best val acc 0.629 (epoch 10)
  
images/8b with reguralization.png

# fixing section 7

seq length T= 10 | RNN acc 0.666 | LSTM acc 1.000
seq length T= 30 | RNN acc 0.086 | LSTM acc 1.000
seq length T= 80 | RNN acc 0.101 | LSTM acc 0.993
seq length T=150 | RNN acc 0.087 | LSTM acc 0.981


