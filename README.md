# ANPR-YOLOV3

Automatic Number Plate Recognition project coded using tensorflow and yolov3 algorithm.

Training currently not supported. Only prediction running(CPU). Running on CPU because, keras.model.predict only one time running for prediction. So it is used as model(data) instead of keras.model.predict(data).

---Pre-trained weights for license plate recognition---

https://drive.google.com/file/d/1JyHaj72pyC2AhIhthV2MzYtqOrY0_yyC/view?usp=sharing

Test Device M1 Macbook Pro : Avarage 4.2 fps on CPU

![test image](https://i.imgur.com/7vTPlM2.png)
