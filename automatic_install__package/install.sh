echo -n "Please enter your password: "
read -s PASSWORD
echo ""

echo ""
echo -n "1. Do you want to set SD_Card Path? (y/N): "
read sd_set


echo -n "2. Do you want to install ROS-melodic? (y/N): "
read ros_install


echo -n "3. Do you want to link opt to SD_Card? (y/N): "
read link_opt 


echo -n "4. Do you want to install IntelRealSense? (y/N): "
read intelRealSense_install 


echo -n "5. Do you want to install openCV? (y/N): "
read openCV_install 


echo -n "6. Do you want to install openVINO 2021V4 LTS? [make sure you are done install IntelRealSense and openCV] (y/N): "
read openVINO_install 


echo -n "7. Do you want run openVino demo? (y/N): "
read openVINO_demo 


echo -n "8. Do you want install tensorflow 2.4.3? (y/N): "
read install_tensorflow 
echo ""

# set SD_CARD_PATH

if [ "$sd_set" '==' "y" ] || [ "$sd_set" '==' "Y" ]; then

	cd /media/$USER/
	export SD_CARD_PATH=/media/$USER/`ls`
	echo "export SD_CARD_PATH=$SD_CARD_PATH" >> ~/.bashrc
	cd $SD_CARD_PATH

else
	echo ""
	echo -n "Skip set SD_Card Path."
	echo ""
fi

echo ""
echo "SD_Card path = $SD_CARD_PATH"
echo ""

# install ROS

echo $PASSWORD | sudo -S apt -y update
echo $PASSWORD | sudo -S apt -y upgrade
echo $PASSWORD | sudo -S apt install -y git

if [ "$ros_install" '==' "y" ] || [ "$ros_install" '==' "Y" ]; then

	git clone https://github.com/CIRCUSPi/ROSKY -b neuron-pi
	echo ""
	cd ~/ROSKY/install_script && echo -e "y\n$PASSWORD" | source ros_install_melodic.sh
	echo ""	
	cd ~/ROSKY/install_script && echo $PASSWORD | source rosky_dependiences.sh
	echo ""
	cd ~/ROSKY/catkin_ws && catkin_make
	echo ""
	cd ~/ROSKY/setup/ && echo $PASSWORD | source set_package_param.sh
	echo ""
	echo "source ~/ROSKY/setup/environment.sh" >> ~/.bashrc
	echo ""
else
	echo ""
	echo -n "Skip install ROS Melodic."
	echo ""
fi

source ~/.bashrc

# link opt to SD_CARD

if [ "$link_opt" '==' "y" ] || [ "$link_opt" '==' "Y" ]; then

	cd /
	sudo mv opt opt_bak 
	cd $SD_CARD_PATH
	sudo mkdir opt
	cd /
	sudo ln -s $SD_CARD_PATH/opt
	cd /opt_bak/
	sudo ln -s /opt_bak/ros /opt

else
	echo ""
	echo -n "Skip link opt to SD_Card."
	echo ""
fi

# IntelRealSense

if [ "$intelRealSense_install" '==' "y" ] || [ "$intelRealSense_install" '==' "Y" ]; then
	
	echo ""
	echo $PASSWORD | sudo apt-get -y update
	echo ""
	echo $PASSWORD | sudo apt-get -y upgrade
	echo ""
	echo $PASSWORD | sudo apt-get -y dist-upgrade
	echo ""
	echo $PASSWORD | sudo apt-get -y install -y git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
	echo ""	
	echo $PASSWORD | sudo apt-get -y install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
	
	cd $SD_CARD_PATH/
	echo ""
	git clone https://github.com/IntelRealSense/librealsense
	echo ""
	cd librealsense/
	echo $PASSWORD | sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
	echo ""
	echo $PASSWORD | sudo udevadm control --reload-rules
	echo ""	
	echo $PASSWORD | sudo udevadm trigger
	echo ""
	echo $PASSWORD | sudo -S ./scripts/patch-realsense-ubuntu-lts.sh
	echo ""
	
	echo ""	
	echo 'hid_sensor_custom' | sudo tee -a /etc/modules
	echo ""
	mkdir build && cd build
	echo ""
	
	echo ""
	cmake ../ -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release
	echo ""
	
	echo ""
	make && echo $PASSWORD | sudo -S make install
	echo ""	
	
	echo ""
	echo $PASSWORD | sudo -S make uninstall && make clean && make && echo $PASSWORD | sudo -S make install
	echo ""
else
	echo ""
	echo -n "Skip install IntelRealSense."
	echo ""
fi	

# openCV

if [ "$openCV_install" '==' "y" ] || [ "$openCV_install" '==' "Y" ]; then
	
	
	echo $PASSWORD | sudo -S apt update
	echo $PASSWORD | sudo -S apt install -y python-pip
	echo $PASSWORD | sudo -S add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
	echo $PASSWORD | sudo -S apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev 
	echo $PASSWORD | sudo -S apt install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper1 libjasper-dev libdc1394-22-dev
	echo $PASSWORD | sudo -S apt install -y python3-pip	
	pip3 install numpy
	pip3 install networkx==2.3

	echo $PASSWORD | sudo -S apt install -y --no-install-recommends libboost-all-dev	
	
	cd $SD_CARD_PATH/
	mkdir -p ./OpenCV3 && cd ./OpenCV3
	
	git clone https://github.com/opencv/opencv.git
	git clone https://github.com/opencv/opencv_contrib.git

	
	cd opencv && git checkout 3.4.2 && cd ..
	cd opencv_contrib && git checkout 3.4.2 && cd ..
	cd opencv

	mkdir build && cd build
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=$SD_CARD_PATH/OpenCV3/opencv_contrib/modules/ ..
	make
	
	echo $PASSWORD | sudo -S make install
	echo "export OpenCV_DIR=$SD_CARD_PATH/OpenCV3/opencv/build" >> ~/.openvino_bashrc

else
	echo ""
	echo -n "Skip install openCV."
	echo ""
fi

# openVINO

if [ "$openVINO_install" '==' "y" ] || [ "$openVINO_install" '==' "Y" ]; then
	
	cd ~/Downloads/
	echo $PASSWORD | sudo -S apt install curl
	curl -O https://registrationcenter-download.intel.com/akdlm/irc_nas/17988/l_openvino_toolkit_p_2021.4.582.tgz
	tar -xvzf l_openvino_toolkit_p_2021.4.582.tgz
	rm l_openvino_toolkit_p_2021.4.582.tgz
	cd l_openvino_toolkit_p_2021.4.582
	echo -e "y" | sudo -S -E ./install_openvino_dependencies.sh
	

	echo ""
	echo ""
	echo ""
	echo ""
	echo "Please follow GUI to install openVINO toolkit, press any key continue.... : "
	read n		
	echo ""
	echo ""
	echo ""
	echo ""
	echo $PASSWORD | sudo -S ./install_GUI.sh
	echo ""
	echo ""
	echo ""
	echo ""

	cd /opt/intel/openvino_2021/install_dependencies
	echo -e "y" | sudo -E ./install_openvino_dependencies.sh
	
	source /opt/intel/openvino_2021/bin/setupvars.sh
	
	echo "source /opt/intel/openvino_2021/bin/setupvars.sh" >> ~/.bashrc
	source ~/.bashrc
	cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites

	echo $PASSWORD | sudo -S ./install_prerequisites_tf2.sh
	pip3 install setuptools 
	echo $PASSWORD | sudo -S apt-get install protobuf-compiler libprotoc-dev

	echo $PASSWORD | sudo -S usermod -a -G users "$(whoami)"
	echo 'SUBSYSTEM=="usb", ATTRS{idProduct}=="2150", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
	SUBSYSTEM=="usb", ATTRS{idProduct}=="2485", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
	SUBSYSTEM=="usb", ATTRS{idProduct}=="f63b", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"' >> 97-myriad-usbboot.rules

	echo $PASSWORD | sudo -S  cp /opt/intel/openvino_2021/inference_engine/external/97-myriad-usbboot.rules /etc/udev/rules.d/

	echo $PASSWORD | sudo -S udevadm control --reload-rules
	echo $PASSWORD | sudo -S udevadm trigger
	echo $PASSWORD | sudo -S ldconfig
	cd /opt/intel/openvino_2021/inference_engine/external/
	python3 -m pip install coverage m2r pyenchant pylint Sphinx safety test-generator


else
	echo ""
	echo -n "Skip install openVINO."
	echo ""
fi

# run OpenVINO demo

if [ "$openVINO_demo" '==' "y" ] || [ "$openVINO_demo" '==' "Y" ]; then

	# install tensorflow 1.5.0
	pip3 list | grep tensorflow
	python3 -m pip install --upgrade --force-reinstall pip
	pip3 show tensorflow
	pip3 install --upgrade tensorflow==1.5.0
	pip install -U PyYaml

	cd /opt/intel/openvino_2021/deployment_tools/demo
	./demo_squeezenet_download_convert_run.sh
	./demo_security_barrier_camera.sh
else
	echo ""
	echo -n "Skip run openVINO demo."
	echo ""
fi

# install tensorflow 2.4.3

if [ "$install_tensorflow" '==' "y" ] || [ "$install_tensorflow" '==' "Y" ]; then

	cd ~/Downloads/	
	
	fileid='1OjtKqdbPKmNoIpL5q-obt_wwdu7oSCeC'
	filename='tensorflow-2.4.3-cp36-cp36m-linux_x86_64.whl'
	
	curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
	curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
	rm ./cookie
	
	pip3 install --user tensorflow-2.4.3-cp36-cp36m-linux_x86_64.whl --force-reinstall

	cd /usr/local/lib/python3.6/dist-packages/tensorflow/core/kernels
	echo $PASSWORD | sudo -S rm libtfkernel_sobol_op.so
	
	cd ~
else
	echo ""
	echo -n "Skip install tendorflow 2.4.3."
	echo ""
fi

echo ""
echo ""
echo ""
echo ""
echo "all done."
echo ""
echo ""
echo ""
echo ""
