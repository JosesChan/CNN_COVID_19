# CNN_COVID_19
 Convolutional neural network used to detect COVID 19 in the lungs, orginally done in Pytorch but has since been converted into Keras.
 Uses samples from the RSNA International COVID-19 Open Radiology Database (RICORD) 1a and 1b datasets
 Model based of one used to detect viral pneumonia.
 
 Requires installation of packages using Python 3.9 (preferably through Anaconda)
 
 Due to the size of Computed Tomography scans, upload to Github is challenging and it is recommended to download either RICORD datasets
 or use a custom dataset with DICOM file format.
 
 Steps to operate:<br/>
 Open environment<br/>
 Download Covid Positive and Negative files from RICORD 1A and RICORD 1B<br/>
 RICORD 1A: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742<br/>
 RICORD 1B: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969771<br/>
 Extract Covid Positive and Negative files<br/>
 Place in the respective folders<br/>
 Run
