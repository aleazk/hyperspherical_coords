# hyperspherical_coords
Python script for transforming Cartesian to Hyperspherical coordinates and back (SC_vectorized.py).

<img width="788" height="763" alt="image" src="https://github.com/user-attachments/assets/d9655ef7-006a-4634-9000-c6f407a72c17" />



We provide here data regarding the differences in the training speeds between the standard VAE and our compression VAE via hyperspherical coordinates. The origin of this difference mainly lies in the extra calculations needed for the coordinate transformations, which are implemented via the script.

The measurements were done during typical trainings in a NVIDIA H100 GPU. We show the results for the case of trainings with CIFAR10, with a batch size of $200$ samples, and the changes in training speed (measured as how many batches per second are being processed) in terms of the dimension $n$ of the latent space. After $n=200$, until $n=800$, the decay is almost linear in $n$, with a decay rate in the speed of $20$ batch/s every $200$ latent dimensions.

<img width="575" height="470" alt="trainspeed (1)" src="https://github.com/user-attachments/assets/9c703c69-41c1-409d-9b6d-e1f529756100" />

