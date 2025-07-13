# hyperspherical_coords
Python script for transforming Cartesian to Hyperspherical coordinates and back (SC_vectorized.py).

![1752436673628-dcb3381d-448c-48a7-9046-edd980728398_1](https://github.com/user-attachments/assets/91d86b29-5b7c-44f7-bb02-c1272f2c10c1)




We provide here data regarding the differences in the training speeds between the standard VAE and our compression VAE via hyperspherical coordinates. The origin of this difference mainly lies in the extra calculations needed for the coordinate transformations, which are implemented via the script.

The measurements were done during typical trainings in a NVIDIA H100 GPU. We show the results for the case of trainings with CIFAR10, with a batch size of $200$ samples, and the changes in training speed (measured as how many batches per second are being processed) in terms of the dimension $n$ of the latent space. After $n=200$, until $n=800$, the decay is almost linear in $n$, with a decay rate in the speed of $20$ batch/s every $200$ latent dimensions.

<img width="575" height="470" alt="trainspeed (1)" src="https://github.com/user-attachments/assets/9c703c69-41c1-409d-9b6d-e1f529756100" />

The script loss_functions.py implements the KLD-like loss for the compression VAE using hyperspherical coordinates, which you can then insert into your VAE as a variation of the standard KLD term, as per the following formulas:

![1752437281033-7264c5cb-38db-417e-9819-6fb6467af7fe_1](https://github.com/user-attachments/assets/101fe225-a51e-4f2b-8d7a-90ac2b434e1f)


For the final KLD-like term you will need to sum all of these, where the gains for each one usually depends on the dataset and architecture. For cifar10 and using a Resnet, we used the following:

kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma

And also a training schedule like the following for the first 100 epochs:

loss = MSE + beta * kl*(epoch**0.5 + 1)
