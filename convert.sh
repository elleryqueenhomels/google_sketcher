mkdir model_js
tensorflowjs_converter --input_format keras ./model/keras_model.h5 ./model_js/
cp categories.txt ./model_js/class_names.txt
zip -r model.zip model_js
