# pt-singan-single-image-gan

[WIP] Inofficial implementation of the paper __"[SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/pdf/1905.01164.pdf)"__ as a project for the _Deep Generative Models_ lecture at TU Darmstadt SS2020.

## Note
- Inner Steps in the Official Implementation?
- Spectral Normalization instead of Gradient Penalty? Other Loss Functions?
- Deformable Convolutions in Generator/Discriminator?


## How to use

For each of the exemplary SinGAN applications, we created an easy-to-use python script that can be run directly from the console by specifying the necessary parameters. All of these scripts have in common that they require either just the run_name of a pretrained SinGAN model or the --not_pretrained flag together with the number of scales N and the number of steps per scale. For instance, the following additional command line arguments would train a SinGAN model with 8 scales and 2000 steps per scale:

```console
python application.py [...] --not_pretrained --N 8 --steps_per_scale 2000
```

If the not_pretrained flag is not given but a trained model with the identifier run_name exists, this is used instead.

### Random Sampling

```console
python sample.py --run_name <String> --img 5026-green-fern-plant-during-daytime.jpg -- height <int> --width <int>
```

### Scale Injection

```console
python scale_injections.py --run_name <String> --img 5026-green-fern-plant-during-daytime.jpg
```

### Super Resolution

```console
python scale_injections.py --run_name <String> --img 5026-green-fern-plant-during-daytime.jpg --super_scales <int>
```

### Paint2Image
_Note_: You have to additionally provide a training image via --img if you want to train a new model. The paint images are expected to be found in the data/paint subdirectory.

```console
python paint2image.py --run_name <String> --paint 5026_1.jpg
```
### Single Image Animation

```console
python animate.py --run_name <String> --frames <int> --fps <int> --alpha 0.1 --beta 0.9 --start_at_scale <int>
```

## Example results

### Random Sampling

training (512x512)         |         512x512           |         512x512           |          512x512          |         512x1024
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![train](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/5026-green-fern-plant-during-daytime.jpg) |![img1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/0_0.jpg) |![img2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/0_1.jpg) |![img3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/0_2.jpg) |![img4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/size_512x1024.jpg) 
![train](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/473-brown-rock-wall.jpg) |![img1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/0_0.jpg) |![img2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/0_1.jpg) |![img3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/0_2.jpg) |![img4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/size_512x1024.jpg) 
![train](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/1497-calm-body-of-water-near-tall-trees-during-daytime.jpg)  |![img1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_1497/0_0.jpg) |![img2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_1497/0_1.jpg) |![img3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_1497/0_2.jpg) |![img4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_1497/size_512x1024.jpg) 

----

### Scale Injections

training (512x512)         |         Scale 0           |         Scale 1           |          Scale 2          |         Scale 3   |         Scale 4
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![train](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/5026-green-fern-plant-during-daytime.jpg) |![img1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_0.jpg) |![img2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_1.jpg) |![img3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_2.jpg) |![img4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_3.jpg) |![img5](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_4.jpg) 

Scale 5         |         Scale 6          |         Scale 7           |          Scale 8          |         Scale 9
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![train](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_4.jpg) |![img1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_6.jpg) |![img2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_7.jpg) |![img3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_8.jpg) |![img4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_9.jpg) |![img5](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_10.jpg) 
