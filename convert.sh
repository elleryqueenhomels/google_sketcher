mkdir model_js
tensorflowjs_converter --input_format keras ./model/keras_model.h5 ./model_js/
mv ./class_names.txt ./model_js/
zip -r model.zip model_js
