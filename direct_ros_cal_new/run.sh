#!/bin/bash
set -x
data="/tmp/data"
bags="/tmp/bags"
output="/tmp/preprocessed"
scripts="/tmp/scripts"

# Fix this befor you run for the fine tuning to not just error!!
ip="192.168.1.7"
export DISPLAY=$ip:0.0

echo "Creating Bags"
makeBags="python3 /tmp/scripts/create_bags.py"
$makeBags
ls $bags
echo "Running preprocessing"
preprocess="ros2 run direct_visual_lidar_calibration preprocess /tmp/bags $output/base -a"
$preprocess


echo "CustomProcessing"
customPreprocess="python3 /tmp/scripts/fix_intensity_images.py"
for i in {0..100..1}
do
    loc="$output/mix$i"
    cp -R $output/base $loc

    customPreprocessFull="$customPreprocess $loc $i"
    $customPreprocessFull

    superglue="ros2 run direct_visual_lidar_calibration find_matches_superglue.py $loc"
    guess="ros2 run direct_visual_lidar_calibration initial_guess_auto $loc"
    fineTune="ros2 run direct_visual_lidar_calibration calibrate $loc --background --auto_quit"
    visualize="python3 /tmp/scripts/visualize_cal.py $loc"

    echo "Finding matches in $loc"
    $superglue
    $guess
    $fineTune
    $visualize
done

echo "Finished!"