# AutoDrive
Use TensorFlow and openCV 
 !semi-finished product!
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
   
