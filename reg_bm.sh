# test_kitti.py: evaluate the performance
python evaluation/test_kitti.py --test_file="./experiments/kitti_10m/test.txt"
python evaluation/test_kitti.py --test_file="./experiments/kitti_loop/test_0_10.txt"
python evaluation/test_kitti.py --test_file="./experiments/kitti_loop/test_10_20.txt"
python evaluation/test_kitti.py --test_file="./experiments/kitti_loop/test_20_30.txt"
python evaluation/test_kitti.py --test_file="./experiments/kitti360_loop/test_0_10.txt"
python evaluation/test_kitti.py --test_file="./experiments/kitti360_loop/test_10_20.txt"
python evaluation/test_kitti.py --test_file="./experiments/kitti360_loop/test_20_30.txt"