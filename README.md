# AutoDrive
Use TensorFlow and openCV 
模拟自动驾驶
semi-finished product!
 Self_ driving at Euro Truck Simulator2
 -----《欧洲卡车模拟2》
screenshoot views (trace the traffic-line):
![screenshoot](https://github.com/YanZiQinKevin/AutoDrive/blob/master/screenshoot/476419097427111374.jpg)
view system for avoiding vehicle
![number "0.5" means distance](https://github.com/YanZiQinKevin/AutoDrive/blob/master/screenshoot/547439799077907042.jpg)


Implementation:
control part: Use win32 api to control keyboard, which self play Euro Truck Simulator2.

Trace the traffic-line part (openCV):

  1, transfer original image to gray  cv2.COLOR_BGR2GRAY
  
  2, than Canny due with the image:
  ![Canny](https://github.com/YanZiQinKevin/AutoDrive/blob/master/screenshoot/truck_Canny.png)
  作用是勾勒路线的轮廓
  
  3， use cv2.HoughLinesP() function to analyze 道路的虚线和实线
         原理是计算条线之间的空隙。而实线比较好处理。
         ![enter image description here](https://github.com/YanZiQinKevin/AutoDrive/blob/master/screenshoot/truck_line.png)
   
 4， Use two lines to figure out "Turn", and the signal will call  win32 to keyboard.


二， Use TensorFlow to recognized front of view, such as Car or Traffic_light. 
      This part follow the :[object_detection](https://github.com/YanZiQinKevin/object_detection)
		
Problem： 通过CNN训练模型，但还是未能解决 红绿信号识别的问题。 
![enter image description here](https://github.com/YanZiQinKevin/AutoDrive/blob/master/screenshoot/truck_light.png)
