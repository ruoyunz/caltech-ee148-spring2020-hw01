# caltech-ee148-spring2020-hw01

Homework #1: Red Light Detection

My simple algorithm for red light detection using matched filtering. 

run_detection contains the main detection algorithm and if run, displays an interactive way of seeing the bounding boxes.

run_predictions will predict the bounding boxes using the detection algorithm in run_detection on the entire dataset, saving it as a json file.

run_visualizations will draw the predictions from the preds.json file and save that image.