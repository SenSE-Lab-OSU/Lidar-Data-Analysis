#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Syntax: $0 [-h|v|V]"
   echo "options:"
   echo "h     Print this Help."
   echo "i     IP for XWinrc server"
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

# Set variables
set -x
data="/tmp/data"
bags="/tmp/bags"
output="/tmp/preprocessed"
scripts="/tmp/scripts"
ip="192.168.1.7"
range=50

############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
while getopts ":hir:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      i) # Enter a name
         ip=$OPTARG;;
      r) # Input a range percent
         range=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

# Running 
export DISPLAY=$ip:0.0


echo "Creating Bags"
makeBags="python3 /tmp/scripts/create_bags.py"
$makeBags
ls $bags
echo "Running preprocessing"
preprocess="ros2 run direct_visual_lidar_calibration preprocess /tmp/bags $output -a"
$preprocess
echo "CustomProcessing"
customPreprocess="python3 /tmp/scripts/fix_intensity_images.py $output $range"
$customPreprocess


superglue="ros2 run direct_visual_lidar_calibration find_matches_superglue.py $output"
guess="ros2 run direct_visual_lidar_calibration initial_guess_auto $output"
fineTune="ros2 run direct_visual_lidar_calibration calibrate $output --background --auto_quit"
visualize="python3 /tmp/scripts/visualize_cal.py $output"

echo "Finding matches in $output"
$superglue
$guess
$fineTune
$visualize

echo "Finished!"