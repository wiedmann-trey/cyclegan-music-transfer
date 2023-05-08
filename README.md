Symbolic Music Style Transfer with CycleGAN and RNNs

- In this project, we built a CycleGAN model to transfer music compositions between three genres: Pop, classical, and jazz. We trained one model to convert jazz songs to classical and vice versa, and we trained another to convert jazz songs to pop and vice versa. We also trained a separate genre classifier to distinguish classical, pop, and jazz songs from one another. Our classifier was trained approximatly 3300 jazz, pop, and classical samples. It reached 90% accuracy on our training data and 79% accuracy on our testing data. 

- The premise of our project was largely inspired by the paper "Symbolic Music Style Transfer with CycleGAN" https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer. We were particularly inspired by their use of the CycleGAN architecture as it eliminated the need for paired data. Our model differs from theirs in that we used PyTorch instead of Tensorflow, and we treated the genre transfer more as a seq-to-seq problem than a pix-to-pix problem. 

In order to preserve the recognizability of the input songs, 