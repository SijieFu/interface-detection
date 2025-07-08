"""
`interface_detection` to detect vials and the liquid interfaces in each vial
in a given image.

The program first uses a pre-trained model to detect vials in the image.
The bounding boxes of the detected vials are then used to crop the vials
out of the image. For each cropped vial, the program applies an edge-based
analysis to find the liquid interface.

This program only contains the edge-based analysis part of the
interface detection process. It is dependent on an external program to
return the bounding boxes of the vials in the image.
"""
