# Bokeh Effect

## Deep Learning Framework for Depth Estimation
We aim to build a robust model that produces bokeh effect from an input image using Deep Learning without using advanced camera lenses or dual lenses. We approach this problem with GANs for depth estimation of input image. We therefore model the Bokeh mode problem to background-foreground seperation (segmentation) problem.
###  NYU Depth Dataset
 For this we used NYU Depth Dataset V2 to train both models. This dataset consists of 1400 images annotated with depth values. We process this depth values to generate depth maps.
 <br>
![enter image description here](https://lh3.googleusercontent.com/gvOy1fJS1P8zSchk5yBWAPB5SKej8Bl0m0r-w0AsuQ8gZx7cwD2keq1q5nota5hJjSPz_omsgycw)![enter image description here](https://lh3.googleusercontent.com/kNSPA7VaOGPiubHXan0Bpyus2RIqhHTvIDY5dEQxcNYk1mVFLIm5KPrOkCceR60qLWudGgm5BI5C)
 ### Pix2Pix
We have trained both Resnet and Unet based Pix2Pix models (but Resnet model is used for final testing). In one step generator is trained twice and discriminator is trained once. It performs very well on NYU Depth Dataset but fails to seperate boundaries for other images (black and white appear in patches). This is possibly because it overfits the dataset (since number of parameters in pix2pix are quite less).
<br>
![enter image description here](https://lh3.googleusercontent.com/QADfPo1O5AFOKMplFnPiSpEPYCNg2Rxzwqy6s8MNTJIMj1clcWO8jsxCV5B9t9fReRj2hPQ4Go77 "Pix2Pix")


<br>

![enter image description here](https://lh3.googleusercontent.com/IuckjCa9rGmslR5ASbi4DcWHfmF3xVO5tU54j4AAu622TifGGLh3bDQj580JgffSRwjfoAH_0nJl)![enter image description here](https://lh3.googleusercontent.com/58KzXHCDNW0vnVtAjOJWF-GAMQcpn9qW5L615AfrPNiSoGXk_lP7NLqcB6YTVxJU9vSXr8bj3t8w)

### Cycle GAN
Cycle-GAN consists of two generators (Resnet Generator) and two discrimantors (Patch-GAN Discriminator). We trained on both 6-block and 9-block generators (9-block generator was found to be better). In one step generator is trained twice and discriminator is trained once. 
Cycle-GAN outperforms Pix2Pix on real world dataset in terms of boundary seperation though it sometimes mis-identifies foreground and background (especially when there is very less color change from background to foreground i.e. low image gradient at the point).
<br>
![enter image description here](https://lh3.googleusercontent.com/OrKT0VSEgIMstHKxzW5DW2JY25HtAJxafb_4XB_3Bcnc8Dw45puzHt_bqKsLqxup_LtzY9ms7DJH)
<br>
![enter image description here](https://lh3.googleusercontent.com/qdlhQewjrfIrrLoniZQ3TG0PVEiDmgEaPF9tvOMYXrBhd_HwRpkGucONzN6yrUYi32aeDTVysT90 "man")![enter image description here](https://lh3.googleusercontent.com/RvHW2aHamC0hCsnfhfg7YxIpO2r-vA2JZNAbs_3ltP_5dz5uDnM1cdx1ily00Eia4fPpm6IFvpcD)



## Results
|Models                |Abs Rel Error                        |RMS Error                       | 
|----------------|-------------------------------|-----------------------------|
|Pix2Pix|      0.2448       |     2.0075    |    
|Cycle-GAN         |0.3728           |3.1207          |   


## Bokeh from Depth + Input 

After depth estimation is completed we use it to generate bokeh effect on input image. For this we heuristically manipulate Depth data to compensate for errors in it to get a good bokeh effect. We apply disc blurring (averaging pixels over a circle) on the input image and use that image to get the background pixels and use original image to get foreground pixels (boundary between foreground and background is typically set between 100-150 for pixel range 0-255). Now for the model to perform better on human images we use opencv implementation of face detection (using haar features) and penalize the regions containing human images to bring them to foreground and prevent them from blurring (in case the model wrongly predicts human faces).

## References

--> https://neurohive.io/en/popular-networks/pix2pix-image-to-image-translation/
--> https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html



