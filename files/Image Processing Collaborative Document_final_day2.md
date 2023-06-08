![](https://i.imgur.com/iywjz8s.png)


# Image Processing Collaborative Document

06-06-2023 Image Processing (day 2).

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](<https://codimd.carpentries.org/s/kCHZi2Iq3>)

Collaborative Document day 1: [link](<https://codimd.carpentries.org/s/6dpg_iY3n>)

Tiny url day 1: https://tinyurl.com/2023-06-05-dc-ip-day1

Collaborative Document day 2: [link](<https://codimd.carpentries.org/s/kCHZi2Iq3>)

Tiny url day 2: https://tinyurl.com/2023-06-06-dc-ip-day2

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website

[link](<https://esciencecenter-digital-skills.github.io/2023-06-05-dc-image-processing/>)

🛠 Setup

[link](<https://github.com/esciencecenter-digital-skills/image-processing/blob/main/setup.md>)

Download files: Data files can be obtained through the repository
[link](https://github.com/esciencecenter-digital-skills/image-processing/tree/main/data)

## 👩‍🏫👩‍💻🎓 Instructors

Djura Smits, Giulia Crocioni, Dani Bodor, Candace Makeda Moore

## 🧑‍🙋 Helpers

Same as above  

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Removed for archiving purposes

## 🗓️ Agenda
| Times Day 1 | Topic                                           |
|:----------- |:----------------------------------------------- |
| 9:30        | Welcome and Intro  (Makeda)                     |
| 9:50        | Image Basics (Makeda)                           |
| 10:10       | Working with Skimage  (Makeda)                  |
| 11:00       | Coffee                                          |
| 11:15       | Drawing (Djura)                                 |
| 12:15       | Lunch                                           |
| 13:15       | Bitwise operations (Djura)                      |
| 14:30       | Creating Histograms (Djura)                     |
| 15:30       | Summary Lecture (Makeda)                        |
| 16:30       | Urgent feedback collection and updates (Makeda) |
| 16:40       | Optional extra tutoring (Makeda)                |
| 17:00       | END                                             |

| Times Day 2 | Topic                                           |
| :---------- |:----------------------------------------------- |
|        9:30 | Welcome back (Makeda)                           |
|        9:40 | Blurring Images (Giulia)                        |
|       10:30 | Coffee break                                    |
|       10:45 | Thresholding Images (Giulia)                    |
|       11:45 | Connected components analysis (Dani)            |
|       12:30 | Lunch break                                     |
|       13:30 | Optional Challenge (Dani)                       |
|       14:30 | Bonus lecture: transformations, affine (Makeda) |
|       16:15 | Feedback collection                             |
|       16:30 | Wine and networking                             |

                                   

 

## 🎓🏢 Evaluation logistics
* At the end of the day you should write evaluations into the colaborative document.


## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .


## 🎓🔧Evaluations

 
Evaluation | specific part or all
 -
 - Your comments| Which session or all sessions
 -
 -
 -
 -
 -
 -
 -
 -
 -
 

## 🔧 Exercises
### Blurring

#### Exercise 1: experimenting with sigma values (10 min)

The size and shape of the kernel used to blur an image can have a significant effect on the result of the blurring and any downstream analysis carried out on the blurred image. Try running the code above with a range of smaller and larger sigma values.



```
img = iio.imread(uri="data/gaussian-original.png")

for s in range(0,6):
    sigma=s
    lowpass = skimage.filters.gaussian(img, sigma=sigma, truncate=1.5, channel_axis=2)
    fig, ax = plt.subplots()
    plt.imshow(lowpass)
```
### Thresholding
#### Exercise 2: more practice with simple thresholding (15 min)

##### First part

Now, it is your turn to practice. Suppose we want to use simple thresholding to select only the coloured shapes (in this particular case we consider grayish to be a colour, too) from the image `data/shapes-02.jpg`.

First, plot the grayscale histogram as in the Creating Histogram episode and examine the distribution of grayscale values in the image. What do you think would be a good value for the threshold t?

#### Second part

Next, create a mask to turn the pixels above the threshold `t` on and pixels below the threshold `t` off. Note that unlike the image with a white background we used above, here the peak for the background colour is at a lower gray level than the shapes. Therefore, change the comparison operator less `<` to greater `>` to create the appropriate mask. Then apply the mask to the image and view the thresholded image. If everything works as it should, your output should show only the coloured shapes on a black background.


#### Example Solution
```python
gray_img = iio.imread(uri="data/shapes-02", mode="L")/255

plot_hist(gray_img, 256, 0.0, 0.1)
```

```python
t = 0.5

binary_mask = gray_img > t

fig, ax = plt.subplots()
plt.imshow(binary_mask)
```

```python
img = iio.imread(uri="data/shapes-02.jpg")
selection =  img.copy()
selection[~binary_mask] = 0

print(selection.shape)

fig, ax = plt.subplots()
plt.imshow(selection)
```

### Connected component analysis
### Exercise 1:

1. Using the `connected_components` function, find two ways of outputting the number of objects found.
2. Does this number correspond with your expectation? Why/why not?
3. Play around with the `sigma` and `thresh` parameters.  
    a. How do these parameters influence the number of objects found?  
    b. OPTIONAL: Can you find a set of parameters that will give you the expected number of objects?

Put your green sticky up when you've finished with 3a


### Exercise 2:

Adjust the `connected_components` function so that it allows `min_area` as an input argument, and only outputs regions (https://scikit-image.org/docs/stable/api/skimage.morphology.html above this minimum.

HINT: check out the [skimage.morphology]l y.

BONUS: explore other morphometrics from skimage.measure.regionprops
1. output the centroid position for each object (above the threshold)
2. consider whether you would export or filter by other properties for your own data and/or in what type of images these could be meaningfull

#### Example Solution
```python=
def connected_components(
    img,
    sigma,
    thresh,
    neighborhood = 2,
    min_area = 0
):
    gray_img = color.rgb2gray(img)
    blurred = filters.gaussian(gray_img, sigma=sigma)
    mask = blurred < thresh
    
       
    mask = morphology.remove_small_objects(mask, min_area)
    
    labeled_img, n_objects = measure.label(mask, 
                                            connectivity=neighborhood,
                                            return_num=True
                                           )

    return labeled_img, n_objects

```
apply the new function
```python=
labeled_img, count = connected_components(shapes, 1, 0.9, min_area=1e5)
show(labeled_img)
print(count)

```


### Capstone Challenge:

In this challenge we will combine a lot of things you have learned over the last few days: blurring, thresholding, masking, connected component analysis, drawing, image formats. 

Write a Python function that uses skimage on images of bacterial colonies to:
- count the number of colonies
- calculate their average area
- calculate the total area of all colonies
- show a new image that highlights the colonies 
- save this image in a lossless format
- try to put your code into a re-usable function, so that it can be applied conveniently to any image file.

Images can be found in the data folder as colonies-01.tif, colonies-02.tif, and colonies-03.tif.


The final output image should look similar to this: 
![](https://hackmd.io/_uploads/r1XXQCoL2.png)

Don't forget to print the number, average area, and total area of colonies for each image.


BONUS (challenging!): on your output image, highlight the most dense colony by drawing a box around it (i.e., lowest mean pixel value).




#### Example Solution
Create overlay
```python=
def output_overlay(
    filename, 
    sigma,
    thresh,
    min_area = 0,
    neighborhood=2):
    # Recreate the connected component analysis
    
    img = iio.imread(filename)
    gray_img = color.rgb2gray(img)
    blurred_img = filters.gaussian(gray_img)
    
    mask = blurred_img < thresh
    
    # remove small objects
    
    mask = morphology.remove_small_objects(mask, min_area, connectivity=neighborhood)
    labeled_img, n_objects = measure.label(mask,
                                           connectivity=neighborhood, 
                                           return_num=True)
    
    clabeled_img = color.label2rgb(labeled_img, bg_label=0)
    
    # 2 overlay on grayscale image
    overlay_img = color.gray2rgb(gray_img)
    overlay_img[mask] = clabeled_img[mask]
    
    print(f"colonies found in {filename} is {n_objects}")
    
    # 3 find the average area
    roi_list = measure.regionprops(labeled_img)
    
    areas = [roi.area for roi in roi_list]
    
    print(f"Average area: {np.mean(areas)}")
    print(f"Total area: {np.sum(areas)}")
    
    
    # BONUS: highlight densest colony on each plate
    roi_list = measure.regionprops(labeled_img, gray_img)
    mean_intensities = [roi.intensity_mean for roi in roi_list]
    min_index = mean_intensities.index(min(mean_intensities))
    bbox = roi_list[min_index].bbox
    
    ry, cx = draw.rectangle_perimeter(start=(bbox[0]-5,bbox[1]-5), end=(bbox[2]+5, bbox[3]+5))
    overlay_img[ry, cx] = 0
    
    
    # Save and show result
    outfile = './' + os.path.basename(filename)[:-4] + '_overlay.png'
    plt.savefig(format='png', fname=outfile)
    show(overlay_img)

    
output_overlay("data/colonies-01.tif", 0.3, 0.3)
output_overlay("data/colonies-02.tif", 0.3, 0.3)
output_overlay("data/colonies-03.tif", 0.3, 0.3)
          
```

## AI and Affine (Bonus Lecture)
### 
Bonus Lecture: (Makeda) AI and Affine

Discussion of preparing images for ML:(as per slides)



Exercise 1/1.5: create code to make new_pic1 and new_pic2 augmented images, that are realistic for a Chest Xray machine learning pipeline, then apply what you percieve as the two most critical algorithms to make them ready for classic supervised machine learning in one bit of code.



## 🧠 Collaborative Notes and Command Log

### Blurring
Blurring makes images smoother.

Blurring is an example of a filtering operation.
You can have high-pass and low-pass filter. A high pass filter will retain the smaller details of the image. A low pass filter retains the larger features.

A convolution is a filter that passes over a vector (1d convolution) or matrix (2d convolution). At every step, the convolution kernel will multiply with the values in the window, and the result will be summed. 

```python
import imageio.v3 as iio
import matplotlib.pyplot as plt
import skimage.filters
%matplotlib widget
```

Read the image
```python
img = iio.imread(uri="data/gaussian-original.png")
```

Display the image
```python
fig, ax = plt.subplots()
plt.imshow(img)
```

Apply the gaussian blur and display
```python
sigma=3.0
blurred = skimage.filters.gaussian(img, sigma=sigma, truncate=3.5, channel_axis=2)
```

Display:
```python
fig, ax = plt.subplots()
plt.imshow(blurred)
```


Asymmetric blurring:
```python
sigma_x=1
sigma_y = 10
blurred = skimage.filters.gaussian(img, sigma=sigma, truncate=3.5, channel_axis=2)
```
```python
fig, ax = plt.subplots()
plt.imshow(blurred)
```

```python
import imageio as iio
import matplotlib.pyplot as plt
import skimage.filters
import numpy as np
import glob
import skimage.color
%matplotlib widget
```
### Thresholding
#### Useful function
```python=
def plot_hist(img, bins, x_min, x_max):
    histogram, bin_edges = np.histogram(img, bins, range=(x_min, x_max))

    fig, ax = plt.subplots()
    plt.plot(bin_edges[0:-1], histogram)
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim(x_min, x_max)
```

#### SImple thresholding
Load a new image

```python
img = iio.imread(uri="data/shapes-01.jpg")
fig, ax = plt.subplots()
plt.imshow(img)
```

Load the image in grayscale:
```python
gray_img = skimage.color.rgb2gray(img)
fig, ax = plt.subplots()
plt.imshow(gray_img, cmap="gray")
```

Blur the grayscale image
```python
blurred_img = skimage.filters.gaussian(gray_img, sigma=1.0)
fig, ax = plt.subplots()
plt.imshow(blurred_img, cmap="gray")
```

Minimum:
```python
blurred_img.min()
```

Maximum
```python
blurred_img.max()
```

Use our function to plot our image histogram:
```python
plot_hist(blurred_img, 256, 0.0, 1.0)
```

Create a mask based on the threshold that we determined from the histogram
```python
t = 0.8
binary_mask = blurred_img < t

fig, ax = plt.subplots()
plt.imshow(binary_mask, cmap="gray")
```

Apply our mask:
```python
# Copy our image to preserve the original
selection = img.copy()
selection[~binary_mask] = 0

fig, ax=plt.subplots()
plt.imshow(selection)
```

Checking dimensions shows that our result is an rgb image
```python=
selection.shape
```

#### Automatic thresholding - Otsu's method
```python
img = iio.imread(uri="data/maize-root-cluster.jpg")
fig, ax=plt.subplots()
plt.imshow(img)
```

```python
gray_image = skimage.color.rgb2gray(img)

blurred_img = skimage.filters.gaussian(gray_image, sigma=1.0)
```

Plot the histogram
```python
plot_hist(blurred_img, 256, 0, 1)
```

Automatic thresholding:
```python
t = skimage.filters.threshold_otsu(blurred_img)

print(f"Found automatic threshold t = {t}")
```

Check if our threshold works
```python
binary_mask = blurred_img > t

fig, ax=plt.subplots()
plt.imshow(binary_mask, cmap="gray")
```

Apply our new mask
```python
selection = img.copy()
selection[~binary_mask] = 0

fig, ax = plt.subplots()
plt.imshow(selection)
```

#### Application: measuring root mass
List all files matching the file pattern
```python
import glob

all_files = glob.glob("data/trial-*.jpg")

# View the list of files
all_files
```

```python=
fig, ax = plt.subplots(1, 4)

fig.tight_layout()

for idx, filename in enumerate(all_files):
    img = iio.imread(uri=filename)
    ax[idx].imshow(img)
```

```python=
def measure_root_mass(filename, sigma):
    img = iio.imread(uri=filename, mode="L")
    
    blurred_img = skimage.filters.gaussian(img, sigma=sigma)
    
    t= skimage.filters.threshold_otsu(blurred_img)
    
    binary_mask = blurred_img > t
    rootPixels = np.count_nonzero(binary_mask)
    
    w = binary_mask.shape[1]
    h = binary_mask.shape[0]

    density = rootPixels / (w * h)
    
    return density
```

Apply to one file
```python
   measure_root_mass(filename="data/trial-016.jpg", sigma=1.5) 
    
```

```python=
all_files = glob.glob("data/trial-*.jpg")

for filename in all_files:
    density = measure_root_mass(filename, sigma=1.5)
    print(filename, density, sep=",")
```

Plotting
```python
fig, ax = plt.subplots(1, 4, figsize=(8, 2))

fig.tight_layout()

for idx, filename in enumerate(all_files):
    img = iio.imread(filename, mode="L")
    
    blurred_img = skimage.filters.gaussian(img, sigma=1.5)
    t = skimage.filters.threshold_otsu(blurred_img)
    binary_mask = blurred_img > t
    ax[idx].imshow(binary_mask, cmap="gray")
```

Plot all the histograms
```python
fig, ax = plt.subplots(4, 3, figsize=(7, 5))

fig.tight_layout()

for idx, filename in enumerate(all_files):
    img = iio.imread(filename, mode="L")
    
    blurred_img = skimage.filters.gaussian(img, sigma=1.5)
    
    # Below comes the new part
    histogram, bin_edges = np.histogram(blurred_img, bins=265, range=(0.0, 1.0))
    t1 = 0.95
    
    binary_mask1 = blurred_img < t1
    selection = blurred_img.copy()
    selection[~binary_mask1] = 0
    t = skimage.filters.threshold_otsu(selection)
    
    binary_mask = selection > t
    
    ax[idx, 0].imshow(blurred_img, cmap="gray")
    ax[idx, 1].plot(bin_edges[0:-1], histogram)
    ax[idx, 2].imshow(binary_mask, cmap="gray")
    
ax[0, 0].title.set_text("blurred image")
ax[0, 1].title.set_text("Blurred image histogram")
ax[0,2].title.set_text("Final binary mask")
```

### Connected component analysis
```python
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from skimage import color, filters, measure, morphology, draw
```

```python
shapes = iio.imread('data/shapes-01.jpg')

def show(img, cmap=None):
    fig, ax = plt.subplots()
    plt.imshow(img, cmap=cmap)
    
show(shapes)
```

Connected component function
```python
def connected_components(
    img,
    sigma,
    thresh,
    neighborhood = 2
):
    gray_img = color.rgb2gray(img)
    blurred = filters.gaussian(gray_img, sigma=sigma)
    mask = blurred < thresh
    
    labeled_img, n_objects = measure.label(mask, 
                                            connectivity=neighborhood,
                                            return_num=True
                                           )
    
    return labeled_img, n_objects
```

Using the function
```python=
labeled_img, n_objects = connected_components(shapes, 2, 0.9)
show(labeled_img, cmap="gray")
```

Give each object a different color
```python=
color_label_img = color.label2rgb(labeled_img, bg_label=0)
show(color_label_img)
```

```python=
roi_list = measure.regionprops(labeled_img)
```

```python=
def get_object_area(roi_list):
    return [roi.area for roi in roi_list]
```

```python=
areas = get_object_area(roi_list)
areas
```

Create histogram to inspect the frequency of different sizes
```python=
fig, ax = plt.subplots()
plt.hist(areas)
```

```python=
min_area = 1e5
large_rois = [roi.area for roi in roi_list if roi.area > min_area]
print(large_rois)

```
### Break until 14:55!

[Post workshop survey](https://carpentries.typeform.com/to/UgVdRQ?slug=2023-06-05-dc-image-processing) 

## Tips & Tops


### Tips (what could be improved)
- having a image processing course 2.0 (more advanced) would be awesome!! (annonymous other person agrees)
- For people with old eyes, switching between screen and laptop is difficult. Better setup a zoom session to share the main screen?
- give more practical examples would be nice, and usuful to understand more in depth the applications. 
- I
- Maybe a bit more example images from research (like the last exercise
- Probably not an issue for most participants, but this course does kind of presume some familiarity with numpy in addition to just python.
- It would be nice if you would start off each section with some information about applicability. it helps people to tune-in a bit more
- I think that the workshop should be restructred. It is ok to have relatively easy things during one entire day. But what we learned just before today's lunch and afternoon was more complicated and fast. Maybe this more complex part can be split into two days? So, we do not do only qick and difficult things during most of the day.
- It would be good to have some material one could work on on parallel once having finished the exercises. 
- Maybe include more exercises along the course optional next steps for people to try different things.Also, introduction to the basics would help. 
- Some basic theory(Mathematics/Image) introduction could be better to understand the Matrix operation of image process.
- More excercises to try at home!
- Extra material for trying at home. 
-Give more explaination about the background and less time for exercise, or leave it for students to figure out at home
- I wanted explanation about image processing for machine learning 
- We did not really focus on the specific applications we wil use this knowledge for, however I think it is hard to do since everyone will use something else.
- make bit more contact with the group after smaller sections of lecture, as to decrease the threshold of asking for clarification (for the students)
- The second day was way more demanding and focused more in things that can be more easily applicabe in research, so it would have been nice to have focused on these during the first day as well.

### Tops (what did you like)
- Nice teachers, give good specific answers to specific questions.
- Good overview on the basic tools for image processing, but also python functions and functionalities...
- Very helpful with questions and problems during the exercises
- Really interesting content. 
- interesting lectures, helpful teachers
- Professional teachers. Very useful for my future work
-   helfpu
- Nice overview of possible options in python to visualize data. 
- very friendly, helpful, and knowledgable intructors!

- Very helpful, dedicated teachers. Interesting material and good combination of theory and scripting. 

-Very helpful and friendly teachers. Even for starters who has little Knowledge, still possible to follow and do some homework.

- The final exercise was challenging and fun
- Many takeaways in how to use the python package.
- Very useful content, enough time for questions, and good staff support
- Really nice and challenging excercises!
- Very good exercises and materials. Very clear application in several fields of research.
- really nice to have these hands-on workshops; of course the knowledge varies very much between people based on their background and expertiese
- I really liked that the workshop was really focussed on practical applications with hands on problems.
## 📚 Resources



You can email (Candace) Makeda to get the cheat sheet if you lost your print-out at c.moore@esciencecenter.nl. 

Popular image format families:
JPEG: https://jpeg.org/
PNG: https://www.w3.org/TR/png/
TIFF: https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf

Hot topics: 
OCR: https://tesseract-ocr.github.io/ is documentation for what is (still) considered the go-to open source library in this area, there are also pre-made software alternatives

Apply for better counting / area measurement:
Watershed - 
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html

StarDist - split of the units with predicting the truth shape
https://pypi.org/project/stardist/



## 🧠📚 Final tips and tops